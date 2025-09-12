import re
import openai
import json
import logging
import os
import paths
import numpy as np
import asyncio
from copy import deepcopy
from collections import Counter
#from lib_api import completion_with_backoff
from async_lib_api import call_multiple_apis, completion_with_backoff
from typing import Dict, List, Optional

from base_dialog_agent import DialogAgent
from priority_consistency_check import PriorityConsistencyChecker
from prompt.prompt_build import prompt_builder
from tools import calculateBestOfferFromLP
from utils import check_null_value, calculate_score, convert_item_cnts_partner, calculate_prob_accept, lower_key_dict, convert_priority_str_to_int, load_json, cache_results, get_cached_results, sigmoid_scale, contains_all_elements, strategy_full_nm_mapper, sync_partner_priorirty_confimation, set_inital_partner_priority, validate_offer


class NegotiationAgent(DialogAgent):
    """GPT Agent base class, later derived to be a AGENT in the scenario
    initial_dialogue_histoy: list of dict. ex) [{"role": "system", "content": "..."}]
    """

    def __init__(self,
                 agent_value_off_table: Dict,
                 initial_dialog_history=None,  #
                 agent_type="", # "negotiator", "partner"
                 engine="gpt-4o-mini",
                 system_instruction=None,
                 args=None,
                 **kwargs
                ):
        """Initialize the agent"""
        super().__init__(initial_dialog_history=initial_dialog_history or [{"role": "system", "content": system_instruction}],
                         agent_type=agent_type,
                         engine=engine,
                         system_instruction=system_instruction
                         )

        logging.debug(f"Initializing {self.agent_type} with engine({self.engine})")
        assert isinstance(agent_value_off_table, dict)

        self.agent_value_off_table = agent_value_off_table
        self.args = args
        self.OSAD_agent = None
        self.preset_partner_priority = kwargs.get('preset_partner_priority', False)
        self.validate_agent_setup()
        self.setup_partner_information()
        self.setup_decision_parameters()
        self.setup_consistency_check_parameters()
        self.initialize_flags()
        self.setup_cached_info()
        self.setup_verbose()
        self.initialize_agent(initial_dialog_history, system_instruction)
        self.priority_consistency_checker = PriorityConsistencyChecker(agent=self, system_instruction=system_instruction, verbose=self.inconsistency_verbose)

    def reset(self):
        self.setup_partner_information()
        self.initialize_flags()
        self.reset_dialog()
        self.setup_decision_parameters()
        self.setup_history_parameters()
        self.setup_consistency_check_parameters()

    def setup_verbose(self):
        self.verbose = self.args.verbose
        self.inconsistency_verbose = self.args.inconsistency_verbose

    def setup_cached_info(self):
        self.lp_results = {}

        if self.lp_caching_on and os.path.exists(paths.CACHED_LP_RESULTS):
            self.lp_results = get_cached_results(paths.CACHED_LP_RESULTS)

    def validate_agent_setup(self):
        """Ensure that the agent's setup is valid."""
        self.engine_STR = self.args.engine_STR # 'gpt-4o'
        assert isinstance(self.agent_value_off_table, dict), "agent_value_off_table must be a dictionary."
        assert all(engine in ["gpt-4-turbo", "gpt-4o", "gpt-4o-mini"] for engine in [self.engine, self.engine_STR]), "Engine and engine_STR must be either 'gpt-4-turbo', 'gpt-4o', or 'gpt-4o-mini'"

    def setup_partner_information(self):
        """Initialize partner priorities and confirmation flags."""
        self.partner_priority = {k: 'null' for k in self.agent_value_off_table.keys()} if not self.preset_partner_priority else deepcopy(self.preset_partner_priority)
        self.priority_confirmation = {k: bool(self.preset_partner_priority) for k in self.agent_value_off_table.keys()}
        self.item_priorities_remaining = {k: {'high': 'true', 'middle': 'true', 'low': 'true'} for k in self.agent_value_off_table.keys()}
        self.partner_offer_history = []
        # logging.info(">> preset_partner_priority: %s", self.preset_partner_priority)
        # logging.info(">> Partner's priority: %s", self.partner_priority)
        # logging.info(">> Partner's priority confirmation: %s", self.priority_confirmation)

    def initialize_flags(self):
        """Initialize flags controlling the agent's behavior in negotiations."""
        self.priority_checker_on = True
        self.priority_asker_on = True
        self.offer_proposer_w_STR = self.args.STR
        self.lp_caching_on = self.args.lp_caching_on
        self.priority_consistency_checker_on = self.args.priority_consistency_checker_on
        self.update_partner_priority_in_checker = self.args.update_partner_priority_in_checker

    def setup_history_parameters(self):
        self.offer_history = []
        self.utterance_offer_history = []
        self.selected_strategy = []
        self.stg_first_step_results = []
        self.generated_params = []
        self.STR3_logs = []

    def setup_consistency_check_parameters(self):
        self.is_partner_priority_consistent = True
        self.inconsistency_detected_case = False
        self.is_counter_offer = False

    def setup_decision_parameters(self):
        """Setup the decision parameters for the agent"""
        self.asking_priority_cnt = 0
        self.tolerance_for_acceptance = 0
        self.partner_offer_met_walkaway = False
        self.compromised_decision_before_end = False
        self.n_OSAD_decision = self.args.n_OSAD_decision
        self.fine_grained_OSAD = self.args.fine_grained_OSAD
        self.n_self_assessment = self.args.n_self_assessment
        self.w1 = self.args.weight_OSAD # weight for acceptance probability
        self.w2 = self.args.weight_self_assessment # weight for assessment score

    def respond(self, user_input):
        """Respond to the user input
        Select the response mode of the three: 1) Ask Preference, 2) Decision (Accept or Walk-away), and 3) Propose Offer
        """

        # Add the user input to the dialog history
        self.dialog_history.append({"role": "user", "content": user_input})
        self.utterance_offer_history.append({"role": "user", "content": self.last_response, "offer": None})

        # Select the response model
        response_mode = self.handle_dialog()
        action, response, agent_offer, partner_offer = response_mode
        self.dialog_history.append({"role": "assistant", "content": response})

        # map utterance and offer
        selected_strategy = self.selected_strategy[-1] if self.selected_strategy and "STR-True" in action  else "None"
        generated_params = self.generated_params[-1] if self.generated_params and action == "STR-True" else None
        STR3_logs = self.STR3_logs[-1] if self.STR3_logs and action == "STR-True" else None
        agent_score, partner_score = self.calucate_score_both(agent_offer)
        self.utterance_offer_history.append({"role": "assistant", "content": response, "offer": agent_offer, 'partner_score':partner_score, 'agent_score': agent_score,'strategy': selected_strategy, 'inferred_partner_priority': self.partner_priority, "gen_params": generated_params, "STR3_logs": STR3_logs})

        return response


    def handle_dialog(self):
        """
        handle_dialog - Ask or propose an offer.
        Based on the current status of the priority confirmation, ask or propose an offer.
        1) ask : if not all items are confirmed, ask the priority.
        2) propose : if all items are confirmed, propose an offer with or without strategic reasoning process.
        """

        self.is_partner_priority_consistent = True # Reset the flag for the consistency of the partner's priority

        #################################################
        # Priority Checker and Update
        #################################################
        if self.priority_checker_on:
            self.check_priority_confirmation_and_updates()
            sync_partner_priorirty_confimation(self.partner_priority, self.priority_confirmation)
            if self.inconsistency_detected_case:
                logging.info(">> Partner's priority is inconsistent. Perform the priority prediction from the STR-1 module")
                self.perform_priority_and_offer_prediction(only_priority_prediction=True)

        # Depending on the status of the priority confirmation, ask or propose an offer
        # Verify both that confirmation is not set and that the priority contains a null.
        if self.priority_asker_on and not self.is_priority_confirmed():
            if self.asking_priority_cnt < 2 : # only ask 2 times
                self.asking_priority_cnt += 1
                return ("ASKING", self.ask_for_priority_confirmation(), None, None)
            else:
                logging.info(f">> Stop asking the partner's priority. Already asked twice. Asking needs or offers, asking cnt:{self.asking_priority_cnt}")
                #self.priority_asker_on = False
                self.asking_priority_cnt = 0
                # for the case where the partner's priority is inconsistent, don't set the partner priroriy to the opposite one
                #if self.is_partner_priority_consistent is True:
                if not self.inconsistency_detected_case: # This variable is updated in the consistency checker
                    self.partner_priority= set_inital_partner_priority(self.partner_priority, self.priority_confirmation, self.agent_value_off_table)
                    logging.info("> Set partner priroity to the opposite one: %s", self.partner_priority)
                else:
                    if check_null_value(self.partner_priority):
                        logging.info(">> After Partner's priority inconsistency is detected, it still has null value in the inferred partner priorty. We will ask the partner what they want")
                        return ("ASKING", self.ask_for_priority_confirmation(ask_for_need_offer=True), None, None)

        #################################################
        # Concsistentcy Checker
        #################################################
        self.priority_asker_on = False
        self.inconsistency_detected_case = False
        if self.priority_consistency_checker_on:
            self.priority_consistency_checker.check_consistency(update_partner_priority=self.update_partner_priority_in_checker)

        if self.priority_consistency_checker_on and not self.is_partner_priority_consistent:
            logging.info(">> Partner's priority is inconsistent!")
            #if self.is_inconsistent_item_confirmed: # Inconsistency for the confirmed items
            logging.info(">> Inconsistency for the confirmed items. Turn on the priority checker and asker")
            self.priority_checker_on = True
            self.priority_asker_on = True
            self.asking_priority_cnt += 1
            self.is_counter_offer = False # counter offers는 없었던 것으로 간주.
            return ("ASKING", self.ask_for_priority_confirmation(), None, None)

        # Add the last offer to the offer history
        if self.is_counter_offer:
            print(">> Partner Counter offer is made. ")
            partner_offer= self.partner_offer_history[-1]
            agent_score, partner_score = self.calucate_score_both(partner_offer)
            self.utterance_offer_history[-1] = {"role": "user", "content": self.last_response, "offer": partner_offer, "agent_score":agent_score, "partner_score":partner_score, "inferred_partner_priority": self.partner_priority}

        return self.make_negotiation_decision()

    def make_negotiation_decision(self):


        ########################################
        # Decision (Accept or Walk-Away)
        ########################################
        # Check if "DEAL" in the last response
        if any(deal.lower() in self.last_response.lower() for deal in ["ACCEPT-DEAL"]):
            logging.debug(">> Partner's last utterance contains 'ACCEPT-DEAL'.")
            return ("ACCEPT-WALKAWAY-DECISON", "ACCEPT-DEAL", None, None)

        # Final Round (When the negotiation is close to the maximum turn), the Agent will make the final decision.
        agent_BATNA = max(self.agent_value_off_table.values()) # top_priority_value = BATNA
        print("compromised_decision_before_end: ", self.compromised_decision_before_end, "| is_counter_offer: ", self.is_counter_offer)
        if self.compromised_decision_before_end and self.is_counter_offer:
            score_from_partner_offer = calculate_score(self.partner_offer_history[-1], self.agent_value_off_table)

            if score_from_partner_offer >= agent_BATNA:  # Accept decision based on the BATNA
                logging.debug(">> Accepting the counter offer: Partner's offer score(%s) >= agent's BATNA(%s)", score_from_partner_offer, agent_BATNA)
                return ("ACCEPT-WALKAWAY-DECISON", "ACCEPT-DEAL", None, None)
            else:
                logging.debug(">> Walking away: Partner's offer score(%s) < agent's BATNA(%s)", score_from_partner_offer, agent_BATNA)
                return ("ACCEPT-WALKAWAY-DECISON", "WALK-AWAY", None, None)

        #####################################
        # Propose an Offer
        #####################################
        generated_response = self.propose_offer(with_ASTRA=self.offer_proposer_w_STR)

        #======== Temp logging ====
        #TEMP: for logging and post analysis. assitant가 아닌 "user"에게  STR 결과를 insert
        self.utterance_offer_history[-1]["gen_params"] = self.generated_params[-1] if self.generated_params else None
        if self.utterance_offer_history[-1]["role"] == "user":
            self.utterance_offer_history[-1]["strategy"] = self.selected_strategy[-1] if self.selected_strategy else None
            self.utterance_offer_history[-1]["STR3_logs"] = self.STR3_logs[-1] if self.STR3_logs else None
        #==========================

        if not self.is_counter_offer:
            return (f"STR-{self.offer_proposer_w_STR}", generated_response, self.offer_history[-1], None)

        ################################
        # Decision (Accept or Walk-Away)
        # Before the Final round, When the partner makes a counter offer, the agent will make a decision based on the counter offer.
        ################################
        score_from_partner_offer = calculate_score(self.partner_offer_history[-1], self.agent_value_off_table)
        STR_selected_offer_score = calculate_score(self.offer_history[-1], self.agent_value_off_table)

        # Accept Condition
        if score_from_partner_offer >= STR_selected_offer_score or self.tolerance_for_acceptance > 2:

            # Accept the partner’s offer only if its score is higher than all previous offers.
            highest_score_from_partner_offer = max([calculate_score(offer, self.agent_value_off_table) for offer in self.partner_offer_history if offer is not None])
            if score_from_partner_offer >= highest_score_from_partner_offer:
                logging.info(">> Accepting the counter offer: Partner's offer score(%s) >= STR selected offer score(%s)", score_from_partner_offer, STR_selected_offer_score)
                return ("ACCEPT-WALKAWAY-DECISON", "ACCEPT-DEAL", None, None)
            else:
                logging.info(">> Not accepting the counter offer: current score from partner's offer (%s) < highest score from partner's offer(%s)", score_from_partner_offer, highest_score_from_partner_offer)
                self.tolerance_for_acceptance += 1

        # Walk-Away Condition: 1) Partner's offer score < agent's BATNA. 2) Partner's offer score does not change in the last three turns.
        if score_from_partner_offer < agent_BATNA:
            logging.info(">> Walk-Away 1st Cond met: Partner's offer score(%s) < value of agent's BATNA(%s)", score_from_partner_offer, agent_BATNA)
            generated_response += "If you keep making offers that only consider your own interests, I'm going to walk away! " # Warning
            if len(self.partner_offer_history) > 1:
                two_turns_ago = self.partner_offer_history[-2]
                if two_turns_ago is None:
                    # case with null value will be skipped
                    pass
                elif len([ x for x in two_turns_ago.values() if x is None or x == 'null']) > 0:
                    # case with null value will be skipped
                    pass
                else:
                    score_partner_prev_offer = calculate_score(self.partner_offer_history[-2], self.agent_value_off_table)
                    if score_from_partner_offer < score_partner_prev_offer:
                        logging.debug(">> Walk-Away 2nd Cond met: Partner's offer score(%s) < previous offer score(%s)", score_from_partner_offer, score_partner_prev_offer)
                        return ("ACCEPT-WALKAWAY-DECISON", "WALK-AWAY", None, None)

        if len(self.partner_offer_history) > 2:

            if self.partner_offer_history[-1] == self.partner_offer_history[-2] != self.partner_offer_history[-3]: # Warning when offers are repeated
                generated_response += "If you keep making offers that only consider your own interests, I'm going to walk away! "
                pass
            elif self.partner_offer_history[-1] == self.partner_offer_history[-2] == self.partner_offer_history[-3]:
                logging.debug(">> Walk-Away 3rd Cond met: Partner's offer is repeated in the last three turns.")
                return ("ACCEPT-WALKAWAY-DECISON", "WALK-AWAY", None, None)

        return (f"STR-{self.offer_proposer_w_STR}", generated_response, self.offer_history[-1], self.partner_offer_history[-1])

    def propose_offer(self, with_ASTRA=True,):
        """Propose an offer with or without strategic reasoning with ASTRA"""
        if with_ASTRA:
            return self.propose_with_ASTRA()
        else:
            return self.propose_without_ASTRA()

    def propose_with_ASTRA(self):
        """Perform strategic reasoning with ASTRA"""

        # Pre-step before ASTRA 3 stages: Predict priorities and Extract the offer
        self.perform_priority_and_offer_prediction()

        ############################################################
        # STAGE 1 - Fairness and Stance Prediction
        ############################################################
        partner_fairness_stance = self.predict_partner_fairness_stance()  # This is first step of ASTRA
        partner_fairness, partner_stance = partner_fairness_stance["partner_fairness"], partner_fairness_stance["partner_stance"]

        ############################################################
        # STAGE 2 -  Generate LP parameters (lambda) and Execute a LPG simulation by using the LP solver to obtain optimal offer candidates
        ############################################################
        LP_query = self.determine_LP_params(partner_fairness, partner_stance)
        self.generated_params.append(LP_query)
        potential_offers = self.simulate_lp_offers(LP_query)

        ############################################################
        # STAGE 3: Select the best offer from with the Partner's Acceptance Probability (PAP) and Strategy Assessment (SA)
        ############################################################
        best_offer = self.select_best_offer(potential_offers)

        # output: Generate a response grounded in the selected offer
        return self.generate_grounded_response(best_offer, with_top_strategy=False)

    def propose_without_ASTRA(self):
        """Propose an offer without strategic reasoning"""
        logging.debug("> [Process] Propose Offer without Strategic Reasoning")

        # prioriy and offer prediction
        self.perform_priority_and_offer_prediction()

        # Generate offer
        best_offer = self.generate_offer()

        # Offer grounded generation
        return self.generate_grounded_response(best_offer, with_top_strategy=False)

    def generate_offer(self):

        round_information = f"{self.cnt_agent_utterance(self.utterance_offer_history)+1} round / {self.args.n_round} rounds" #if self.utterance_offer_history else f"1 round / {self.args.n_round}  rounds"
        proc_dialog_history = self.processed_dialog_history()
        proc_offer_history = self.process_utter_offer_history(self.utterance_offer_history, self.int_partner_priority, w_strategy=True, w_utterance=False, role_user="PARTNER OFFER", role_assistant="YOUR OFFER")
        proc_concession_history = self.process_utter_offer_history(self.utterance_offer_history, self.int_partner_priority, w_utterance=False, set_turn_index=True, filter_None_offer=True, role='user')

        logging.info("\n======== offer_history ===== \n%s", self.process_utter_offer_history(self.utterance_offer_history, self.int_partner_priority, w_strategy=True, w_utterance=False, w_inferred_partner_priority=True, w_lp_params=True, role_user="PARTNER OFFER", role_assistant="YOUR OFFER"))
        logging.info("\n======== concession_history ===== \n%s", self.process_utter_offer_history(self.utterance_offer_history, self.int_partner_priority, w_utterance=False, set_turn_index=True,  w_inferred_partner_priority=True, w_lp_params=True,filter_None_offer=True, role='user'))

        prompt=prompt_builder(self.agent_value_off_table, None, None, proc_dialog_history, offer_history=proc_offer_history, concession_history=proc_concession_history, expert_persona='all', prompt_type='generate_offer', round=round_information, verbose=False)

        msg= {"messages": [{ "role": "user", "content": prompt}], "model": self.engine_STR, "json_parsing_check": True}
        generated_offer = None
        while not validate_offer(generated_offer):
            logging.info("> Generating and Validating Offer..")
            raw_response = self.call_engine(**msg)
            response = json.loads(raw_response["content"])
            generated_offer, seleted_strategy = response['offer'], response.get('strategy')

        if seleted_strategy:
            self.selected_strategy.append(strategy_full_nm_mapper(seleted_strategy))
        else:
            self.selected_strategy.append("None")
        score = calculate_score(generated_offer, self.agent_value_off_table)
        logging.debug(">> Finally Selected Best Offer: [Score %s] %s" , score, generated_offer)

        # Add the selected offer to the offer history
        self.offer_history.append(generated_offer)
        return [score, generated_offer['food'], generated_offer['water'], generated_offer['firewood']]

    def determine_LP_params(self, partner_fairness, partner_stance, **kwargs):
        """ Determine LP parameters"""

        round_information = f"{self.cnt_agent_utterance(self.utterance_offer_history)+1} round / {self.args.n_round} rounds" #if self.utterance_offer_history else f"1 round / {self.args.n_round}  rounds"
        proc_dialog_history = self.processed_dialog_history()
        proc_offer_history = self.process_utter_offer_history(self.utterance_offer_history, self.int_partner_priority, w_strategy=True, w_utterance=False, role_user="PARTNER OFFER", role_assistant="YOUR OFFER")
        proc_concession_history = self.process_utter_offer_history(self.utterance_offer_history, self.int_partner_priority, w_utterance=False, set_turn_index=True, filter_None_offer=True, role='user')
        latest_offer = self.offer_history[-1] if self.offer_history else ""
        latest_offer_str = f"Score={calculate_score(latest_offer, self.agent_value_off_table)}: Food={latest_offer['food']}, Water={latest_offer['water']}, Firewood={latest_offer['firewood']}" if latest_offer else "No latest offer suggested"

        logging.info("\n======== offer_history ===== \n%s", self.process_utter_offer_history(self.utterance_offer_history, self.int_partner_priority, w_strategy=True, w_utterance=False, w_inferred_partner_priority=True, w_lp_params=True, role_user="PARTNER OFFER", role_assistant="YOUR OFFER"))
        logging.info("\n======== concession_history ===== \n%s", self.process_utter_offer_history(self.utterance_offer_history, self.int_partner_priority, w_utterance=False, set_turn_index=True,  w_inferred_partner_priority=True, w_lp_params=True,filter_None_offer=True, role='user'))

        prompt=prompt_builder(self.agent_value_off_table, None, None, proc_dialog_history, offer_history=proc_offer_history, concession_history=proc_concession_history, expert_persona='all', prompt_type='generate_LP_param_max_lambda', round=round_information, latest_offer=latest_offer_str, partner_fairness_stance=partner_fairness_stance, verbose=False)

        #print("> Generate LP questions\n", prompt)
        #exit()
        msg= {"messages": [{ "role": "user", "content": prompt}], "model": self.engine_STR, "json_parsing_check": True}
        raw_response = self.call_engine(**msg)
        response = json.loads(raw_response["content"])

        #logging.info("Generated LP parameters: %s", json.dumps(response, indent=4))
        #lp_rationale, lp_max, lp_lambda, lp_fairness, lp_stance = response['rationale'], response['max_bound'], response['lambda'], response['partner_offer_fairness'], response['partner_stance']
        lp_rationale, lp_max, lp_lambda = response['rationale'], response['max_bound'], response['lambda']

        logging.info("-"*50)
        logging.info("> #### LP parameters: max_bound=%s, lambda=%s, fairness=%s, stance=%s #######", lp_max, lp_lambda, partner_fairness, partner_stance)
        logging.info("> LP rationale: %s", lp_rationale)

        arg_lambda= response.get("lambda") or None
        arg_max_bound = response.get("max_bound") or None
        p_offer_fairness = partner_fairness #response.get("partner_offer_fairness") or None
        p_stance = partner_stance #response.get("partner_stance") or None
        self.stg_first_step_results.append({"lambda": arg_lambda, "max_bound": arg_max_bound, "partner_fairness": p_offer_fairness, "partner_stance": p_stance, 'rationale': lp_rationale})

        return {"lambda": arg_lambda, "max_bound": arg_max_bound, "partner_fairness": p_offer_fairness, "partner_stance": p_stance}

    def predict_partner_fairness_stance(self):
        """ Determine LP parameters"""
        #round_information = f"{self.current_dialogue_round} round / {self.args.n_round} rounds"
        round_information = f"{self.cnt_agent_utterance(self.utterance_offer_history)+1} round / {self.args.n_round} rounds" #if self.utterance_offer_history else f"1 round / {self.args.n_round}  rounds"
        proc_dialog_history = self.processed_dialog_history()
        proc_offer_history = self.process_utter_offer_history(self.utterance_offer_history, self.int_partner_priority, w_strategy=True, w_utterance=False, role_user="PARTNER OFFER", role_assistant="YOUR OFFER", num_of_latest_turns=1)
        proc_concession_history = self.process_utter_offer_history(self.utterance_offer_history, self.int_partner_priority, w_utterance=False, set_turn_index=True, filter_None_offer=True, role='user', num_of_latest_turns=2)

        #logging.info("\n======== offer_history ===== \n%s", self.process_utter_offer_history(self.utterance_offer_history, self.int_partner_priority, w_strategy=True, w_utterance=False, w_inferred_partner_priority=True, w_lp_params=True, role_user="PARTNER OFFER", role_assistant="YOUR OFFER"))
        #logging.info("\n======== concession_history ===== \n%s", self.process_utter_offer_history(self.utterance_offer_history, self.int_partner_priority, w_utterance=False, set_turn_index=True,  w_inferred_partner_priority=True, w_lp_params=True,filter_None_offer=True, role='user'))

        prompt_fairness=prompt_builder(self.agent_value_off_table, None, None, proc_dialog_history, offer_history=proc_offer_history, expert_persona='all', prompt_type='generate_LP_param_fairness', verbose=False)
        prompt_stance= prompt_builder(self.agent_value_off_table, None, None, proc_dialog_history, concession_history=proc_concession_history, expert_persona='all', prompt_type='generate_LP_param_stance', verbose=False)

        msg_fairness= {"messages": [{ "role": "user", "content": prompt_fairness}], "model": self.engine_STR, "json_parsing_check": True}
        msg_stance= {"messages": [{ "role": "user", "content": prompt_stance}], "model": self.engine_STR, "json_parsing_check": True}
        fair_response = self.call_engine(**msg_fairness)
        stance_response = self.call_engine(**msg_stance)
        fair_response = json.loads(fair_response["content"])
        stance_response = json.loads(stance_response["content"])
        fairness = fair_response.get("partner_offer_fairness")
        stance = stance_response.get("partner_stance")
        fair_rationale = fair_response.get("rationale_for_fairness")
        stance_rationale = stance_response.get("rationale_for_stance")

        logging.info("-"*50)
        logging.info("> #### LP parameters: fairness=%s, stance=%s #######", fairness, stance)
        logging.info("> LP rationale for fairness: %s", fair_rationale)
        logging.info("> LP rationale for stance: %s", stance_rationale)
        logging.info("-"*50)
        return {"partner_fairness": fairness, "partner_stance": stance}


    def simulate_lp_offers(self, query=None):
        """Second step of strategic reasoning: Generate LP parameters and perform simulation."""

        lambda_LP= query["lambda"]
        maximum_val_LP = query["max_bound"]

        score_of_latest_offer = sum(self.agent_value_off_table[item] * self.offer_history[-1][item] for item in self.agent_value_off_table) if self.offer_history else 36
        #assert maximum_val_LP <= score_of_latest_offer, "The maximum value in LP should be less than the score of the latest offer."
        if int(maximum_val_LP) > score_of_latest_offer:
            logging.info(f">> Issue with the generated the max bound {maximum_val_LP} (> the score of latest offer({score_of_latest_offer})). Setting it by rule (score_of_latest_offer)")
            maximum_val_LP = score_of_latest_offer

        if not (maximum_val_LP > 10 and maximum_val_LP <= 36) or maximum_val_LP is None:
            logging.info(f">> Issue with the generated the max bound {maximum_val_LP}. Setting it by rule")
            maximum_val_LP = self.set_maximum_value_for_LP()  # Set the maximum value in LP equation

        logging.info(">> Maximum value for LP: %s", maximum_val_LP)

        # check LP cache
        lp_cache_key = (maximum_val_LP, tuple(self.agent_value_off_table.items()), tuple(self.int_partner_priority.items()))
        if self.lp_caching_on and lp_cache_key in self.lp_results:
            logging.info('>> LP results: found cached results')
            return sorted(self.lp_results[lp_cache_key], key=lambda x: x[0], reverse=True)[:min(top_n, len(self.lp_results))]
        logging.debug('>> LP results: no cached results')

        # Excute the LP function to generate potential offers
        try:
            logging.debug(">> Executing LP function with the parameters.: max_value(%s), partner_priority(%s)", maximum_val_LP, self.int_partner_priority)
            offer_list = calculateBestOfferFromLP(maximum_val_LP, lambda_lp, self.agent_value_off_table, self.int_partner_priority)
            self.cache_and_update_lp_results(lp_cache_key, offer_list)
        except Exception as e:
            logging.error(f"Error executing LP function: {e}")
            logging.error(f">> the parameters used: maximum_val_LP:{maximum_val_LP} | agent_value_off_table:{self.agent_value_off_table} | int_partner_priority:{self.int_partner_priority}")
            return []

        # Check if the LP function returned any offers
        if not offer_list:
            logging.error(f"No offers from LP. Please check the parameters for the LP function. maximum_val_LP:{maximum_val_LP} | agent_value_off_table:{self.agent_value_off_table} | int_partner_priority:{self.int_partner_priority}")
            return []

        # Sort offers by their total score in descending order
        sorted_offers = sorted(offer_list, key=lambda x: x[0], reverse=True)

        # Check and filter the offers : filter out 1) the offers with the same offer in previous & 2) the offer not beneficail to both players
        _previous_offer = self.offer_history[-1] if self.offer_history else None
        if _previous_offer:
            # 1) Filter out the offer with the same offer in the previous offer
            #_sorted_offers = [offer for offer in sorted_offers if offer[1:] != (_previous_offer['food'], _previous_offer['water'], _previous_offer['firewood'])]
            _sorted_offers = [offer for offer in sorted_offers]

            # 2) Filter out offers not beneficial to both players
            agent_score_prev = calculate_score(_previous_offer, self.agent_value_off_table)
            partner_score_prev = convert_item_cnts_partner(_previous_offer, self.int_partner_priority)[0]
            _sorted_offers = [offer for offer in _sorted_offers if offer[0] >= agent_score_prev or convert_item_cnts_partner(offer, self.int_partner_priority)[0] >= partner_score_prev]


            #logging.info(">> LP offers after filtering: from %s to %s", len(sorted_offers), len(_sorted_offers))
            sorted_offers = _sorted_offers

        _previous_offer = self.offer_history[-1] if self.offer_history else None

        if not sorted_offers:
            logging.error("All sorted offers were filtered out. No offers to return.")
            return []

        # Print sorted offers
        logging.debug(">> Offer list from LP:")
        for offer in sorted_offers:
            logging.debug(f"[Total Score: {round(offer[0])}] Food:{int(offer[1])}, Water:{int(offer[2])}, Firewood:{int(offer[3])}")

        # Return the top 6 offers, assuming there are at least 6 offers
        top_offers = sorted_offers[:min(self.args.top_n, len(sorted_offers))]
        return top_offers

    def select_best_offer(self, offers):
        """Third step of strategic reasoning: Minimize regret by selecting the most strategic offer."""
        assert offers, "No offers from LP. Please check the parameters for the LP function."

        selected_offer, selected_stg = self.strategic_reasoning_third_stg(offers)

        # Update the offer history with the selected or fallback offer
        if selected_offer:
            offer_details = dict(zip(['food', 'water', 'firewood'], selected_offer[1:]))
            self.selected_strategy.append(strategy_full_nm_mapper(selected_stg))
            logging.debug(">> Finally Selected Best Offer: [Score %s] %s" , selected_offer[0], offer_details)
        else:
            logging.debug(">> No suitable offer provided from LP. Using the previous offer.")
            offer_details = self.previous_offer  # Fallback to the previous offer if no suitable one found
            self.selected_strategy.append("None")

        # Add the selected offer to the offer history
        self.offer_history.append(offer_details)
        return selected_offer

    def generate_grounded_response(self, offer, with_top_strategy=False):
        """Generate a grounded response based on the selected offer, with optional strategy inclusion."""
        offer_details = f"Food:{offer[1]}, Water:{offer[2]}, Firewood:{offer[3]}" if offer else "No offer selected"
        response = self.offer_grounded_generation(offer_details)

        return f"[{self.selected_strategy[-1]}] {response}" if with_top_strategy else response

    def double_check_partner_offer(self, n, dialog_history):
        prompt = prompt_builder(agent_value_off_table=None, partner_inferred_priority=None,
                                priority_confirmation=None,
                                conversation_history=dialog_history,
                                prompt_type="double_check_partner_offer",
                                verbose=False)

        msg = {"messages": [{"role": "user", "content": prompt}], "n": n, "json_parsing_check": True}
        message = self.call_engine(**msg)
        message = [message] if n == 1 else message

        json_list = [json.loads(msg['content'])['items_partner_take'] for msg in message]
        candidates = [frozenset(d.items()) for d in json_list]
        most_common_candidates = Counter(candidates).most_common()
        return dict(most_common_candidates[0][0]) if most_common_candidates else None

    def perform_priority_and_offer_prediction(self, only_priority_prediction=False, **kwargs):

        """First step of Strategic Reasoning based on dialog history
        - Priority prediction
        - Offer prediction
        """
        #################
        # 1) Quesstion about partner's priority and offer
        # For the priority prediction, previous partner's priority information and whole conversation are required.
        # For the offer prediction, it can be done at utterance level. But currently, we are using the whole conversation history.
        ################
        # Load the initial instruction for partner's priroriry
        assert self.agent_value_off_table is not None
        assert self.dialog_history[-1]['role'] == 'user'

        logging.debug("> [Process] Strategic Reasoning First Stage")
        agent_value_off_table = self.agent_value_off_table
        partner_priority = self.partner_priority
        priority_confirmation = self.priority_confirmation

        ####################################
        # Partner's Priority prediction
        ####################################
        proc_dialog_history = self.processed_dialog_history()
        priority_q=prompt_builder(agent_value_off_table, partner_priority, priority_confirmation, proc_dialog_history, prompt_type='priority', verbose=self.verbose)
        priority_response=self.call_engine(messages=[{ "role": "user", "content": priority_q}],  json_parsing_check=True)
        #logging.debug("priority response : %s", priority_response))
        _priority_response = json.loads(priority_response["content"])
        #logging.debug("_priority_response: %s", _priority_response))
        # temporary unmark
        if not self.priority_asker_on or only_priority_prediction:  # Asking을 하지 않을때 또는 Predition module만 켜져 있을때 업데이트.
            logging.info(">> [STG-1] Priority Update Process....")
            inferred_partner_priority = lower_key_dict(_priority_response['Q1']["Answer"])
            partner_prioriry_to_be_updated = lower_key_dict(_priority_response['Q2']["Answer"])
            #logging.debug("<priority prediction prompt> \n%s", priority_q))
            #logging.debug(">> inferred_partner_priority : %s", inferred_partner_priority)
            # Update partner's priority

            # Update only if it’s unconfirmed and the priority has a null value.
            if not self.is_priority_confirmed() and check_null_value(partner_priority):
                partner_prioriry_to_be_updated=self.update_partner_priority(partner_prioriry_to_be_updated)

                # update partner's priority
                if self.partner_priority != partner_prioriry_to_be_updated:
                    logging.info("\n%s\n[STG-1] Partner's priority will be updated from (Current) %s to (Updated) %s\n%s\n", "*"*50, self.partner_priority, partner_prioriry_to_be_updated, "*"*50)
                    self.partner_priority = partner_prioriry_to_be_updated
                    print(self.utterance_offer_history, self.partner_offer_history, self.partner_priority)
                    if self.partner_offer_history:
                        if self.utterance_offer_history:
                            self.utterance_offer_history[-1] = {"role": "user", "content": self.last_response, "offer": self.partner_offer_history[-1], "inferred_partner_priority": self.partner_priority}
                        else:
                            self.utterance_offer_history.append({"role": "user", "content": self.last_response, "offer": self.partner_offer_history[-1], "inferred_partner_priority": self.partner_priority})

            # check null value in partner's priority

        if only_priority_prediction:
            return

        #########################
        # Partner Offer Extraction
        #########################
        proc_dialog_history = self.processed_dialog_history(num_of_latest_turns=2)
        offer_q=prompt_builder(agent_value_off_table, partner_priority, priority_confirmation, proc_dialog_history, prompt_type='offer', verbose=self.verbose)
        offer_response=self.call_engine(messages=[{ "role": "user", "content": offer_q}],  json_parsing_check=True)
        partner_offer = json.loads(offer_response["content"])['Q1']["Answer"]
        partner_offer = lower_key_dict(partner_offer)

        # check "null" in the partner_offer
        #self.is_counter_offer = False
        # Todo: This should be replaced by the consistency checker
        if self.priority_consistency_checker_on:
            if self.is_counter_offer:
                #assert self.partner_offer_history[-1] == partner_offer, "The counter offer from the consistency checker is not consistent with the one from STR"
                # Todo: later, we will assert the consistency between the counter offer from the consistency checker and the one from STR
                if self.partner_offer_history[-1] != partner_offer:
                    logging.critical(
                        "!! Offer (CC) != Offer (STR). Comparing offer between the Consistency Checker (CC) and STR 1st stg\n"
                        f"partner offer from CC : {self.partner_offer_history[-1]}\n"
                        f"partner offer from STR : {partner_offer}")

                    _dialog_history = self.processed_dialog_history(num_of_latest_turns=1)
                    double_checked_final_offer = self.double_check_partner_offer(n=1, dialog_history=_dialog_history)

                    # Correct partner offer history after double-checking
                    #self.partner_offer_history[-1] = convert_item_cnts_partner(double_checked_final_offer, only_cnts=True)
                    reextracted_partner_offer = convert_item_cnts_partner(double_checked_final_offer, only_cnts=True)

                    #TEMP : Change the partner_offer to the one from the consistency checker into one from STR
                    if reextracted_partner_offer == self.partner_offer_history[-1]:
                        logging.info(">> Keep the parter offer from CC")
                        pass
                    elif reextracted_partner_offer == partner_offer:
                        self.partner_offer_history[-1] = partner_offer
                        logging.info(">> Change the partner_offer to the one from STR given reextracted offer is the same as the one from STR")
                        logging.info(">> Change the partner_offer to the one from the consistency checker into one from STR")
                        logging.info(">> partner_offer_history: %s", self.partner_offer_history)
                    else:
                        logging.error(f">> reextraced offer is not same as both ones from CC and STR: {reextracted_partner_offer}")

                    if self.utterance_offer_history[-1]["role"] == "user":
                        self.utterance_offer_history[-1]["offer"] = self.partner_offer_history[-1]
                        logging.info(">> Change the partner_offer in the utterance_offer_history")

        else: # Wo CC, we update counter offer from the first STR Process
            self.is_counter_offer = False
            if not check_null_value(partner_offer):
                self.is_counter_offer = True
                self.partner_offer_history.append(partner_offer)

            #logging.debug("<offer prediction prompt> \n", offer_q))
            logging.debug(">> predicted partner_offer from STR-1 : %s", partner_offer)


    def check_score_repetition(self, check_turns=2):
        if len(self.offer_history) < check_turns:
            return (False, 0)

        previous_score = calculate_score(self.offer_history[-1], self.agent_value_off_table)
        for i in range(1, check_turns):
            if calculate_score(self.offer_history[-i], self.agent_value_off_table) != previous_score:
                return (False, previous_score)
        return (True, previous_score)


    def strategic_reasoning_third_stg(self, offers_from_LP):
        """Minimize regret"""
        logging.debug("> [Process] Strategic Reasoning Third Stage: Minimize Regret")

        assert self.OSAD_agent is not None, "OSAD_agent should be provided for one step ahead decision"

        self.OSAD_agent.setup_value_off_table(self.int_partner_priority) # Set up OSAD_agent's value offer table with partner's priority

        osad_msgs, sa_msgs, offer_fineg_assessment = [], [], []

        # Process offers and generate processed offer candidates string
        processed_offer_candidates = []
        for idx, offer in enumerate(offers_from_LP):
            partner_score_items = convert_item_cnts_partner(offer, self.int_partner_priority)
            processed_offer_candidates.append(
                f"[Offer-index {idx}] Score={partner_score_items[0]}"
            )
        processed_offer_candidates_str = "\n".join(processed_offer_candidates)

        # Evaluate each offer and generate PAP messages
        partner_scores_list=[]
        processed_agent_offer_candidates= []
        for offer_idx, offer in enumerate(offers_from_LP):
            partner_score_items = convert_item_cnts_partner(offer, self.int_partner_priority)
            suggested_offer_for_partner = (
                f"[Score {partner_score_items[0]}] Food: {partner_score_items[1]}, "
                f"Water: {partner_score_items[2]}, Firewood: {partner_score_items[3]}"
            )
            partner_scores_list.append(partner_score_items[0])


            # Fine-grained assessment of the offer
            osad_msgs.append(
                self.OSAD_agent.fine_grained_osad(
                    self.int_partner_priority,
                    suggested_offer_for_partner,
                    offer,
                    processed_offer_candidates_str,
                    self.dialog_history,
                    number_of_assessment=self.n_OSAD_decision,
                    only_return_msg=True
                )
            )


            processed_agent_offer_candidates.append(
                f"[Offer Candidate {offer_idx}] Score={offer[0]}: Food={offer[1]}, Water={offer[2]}, Firewood={offer[3]}"
            )
        processed_agent_offer_candidates_str = "\n".join(processed_agent_offer_candidates)
        sa_msgs.append(self.assess_offer(None, offer_candidates=processed_agent_offer_candidates_str, number_of_assessment=self.n_self_assessment, only_return_msg=True))


        # Asynchronous call to the engine
        combined_api_calls = osad_msgs + sa_msgs
        loop = asyncio.get_event_loop()
        combined_results = loop.run_until_complete(call_multiple_apis(combined_api_calls))
        results1 = combined_results[:len(osad_msgs)]
        results2 = combined_results[len(osad_msgs):]


        assert all(isinstance(x, dict) for x in combined_results), f"API results (OSAD + Assement) should be a list of dictionaries. combined_results: {combined_results}"
        if not all(isinstance(x, dict) for x in combined_results):
            error_case = [ x  for x in combined_results if not isinstance(x, dict)]
            logging.error(f"!API results (OSAD + Assement) Error case: {error_case}")

        # Processing the results
        # OSAD Results
        total_score, stg_potential=  self.process_fg_osad_results(results1)

        # SA results
        top_2_stgs, top_offer, rationales, offer_ranks = self.process_fg_sa_results3(results2, number_of_offers=len(offers_from_LP))

        top_stgs_str = ",".join([ f"{x[0]}({x[1]})" for x in top_2_stgs])
        top_stg = top_2_stgs[0][0]
        print("Top 2 Stgs: ", top_2_stgs)
        #self.selected_strategy.append(top_stg)
        # got top offer from the offer_ranks
        #print("top_offer: ", top_offer)
        #print("offer_ranks: ", offer_ranks)
        if top_offer != offers_from_LP[offer_ranks.index(0)]:
            logging.error(f"Top offer from the offer_ranks should be the same as the top_offer {top_offer} | {offers_from_LP[offer_ranks.index(0)]}")


        ranked_offers = [value for rank, value in  sorted(zip(offer_ranks, offers_from_LP))]

        # ranking normalization from 0 to 1
        min_rank, max_rank = min(offer_ranks), max(offer_ranks)
        normalized_ranks = [(max_rank - rank) / (max_rank - min_rank) for rank in offer_ranks]

        # Fine-grained OSAD Results
        fg_w = [0.7, 0.3]
        fg_scores = list(zip(offers_from_LP, total_score, stg_potential, normalized_ranks, partner_scores_list))
        final_scores = [(offer, round(fg_w[0]*ts + fg_w[1]*sp, 2), round(ts,2), round(sp, 2), round(norm_rank, 2), round(PS,3)) for offer,  ts, sp , norm_rank , PS in fg_scores]
        final_scores = [ (x[0], round(self.w1*x[1] + self.w2*x[4], 2) , x[1],x[2],x[3],x[4],x[5])for x in final_scores]
        final_score_dict= {"offer_candidates":{}, "best_offers":{}}

        for idx, (offer, final_score,  wegithed_fg_OSAD, ts, sp, norm_rank, PS) in enumerate(final_scores):
            logging.info(f">>> Offer (%s): (%s: f=%s w=%s fw=%s) | Final W. Score: %s | OSAD: %s (TS: %s / SP: %s) | SA-Norm_Rank: %s | Partner Score: %s", idx, offer[0], offer[1], offer[2], offer[3], final_score, wegithed_fg_OSAD, ts, sp, norm_rank, PS)
            final_score_dict["offer_candidates"][idx] = {"Offer": offer, "Final_Score": final_score, "PAP": wegithed_fg_OSAD, "TS": ts, "SP": sp, "Norm_Rank": norm_rank, "PartScore": PS}
        logging.info("---")


        # print Rationales for STR3
        logging.info(" Top Stg: %s | Top Offer: %s | Rationales: \n%s", top_stgs_str, top_offer, "\n".join(f"{index}: {item}" for index, item in enumerate(rationales, 1) if top_stg in item))

        #TEMP ablation study for STR3
        #best_offer = offers_from_LP[offer_ranks.index(0)] #top_offer
        best_offer, best_offer_Score = (max(final_scores, key=lambda x: x[1])[0], max(final_scores, key=lambda x: x[1])[1]) if final_scores else (None, None)
        best_offer_only_PAP, best_offer_only_PAP_Score = (max(final_scores, key=lambda x: x[2])[0], max(final_scores, key=lambda x: x[2])[1]) if final_scores else (None, None)
        best_offer_only_SA, best_offer_only_SA_Score = (max(final_scores, key=lambda x: x[5])[0], max(final_scores, key=lambda x: x[5])[1]) if final_scores else (None, None)

        final_score_dict["best_offers"]["best_offer"] = {"Offer": best_offer, "Final_Score": best_offer_Score}
        final_score_dict["best_offers"]["best_offer_only_PAP"] = {"Offer": best_offer_only_PAP, "Final_Score": best_offer_only_PAP_Score}
        final_score_dict["best_offers"]["best_offer_only_SA"] = {"Offer": best_offer_only_SA, "Final_Score": best_offer_only_SA_Score}
        final_score_dict["stg_rationales"] = {index: item for index, item in enumerate(rationales) if top_stg in item}

        #TEMP
        self.STR3_logs.append(final_score_dict)


        _previous_offer = self.offer_history[-1] if self.offer_history else None
        if top_stg not in ["NCR", "RNC"]:
            # 1) Filter out the offer with the same offer in the previous offer
            #_sorted_offers = [offer for offer in sorted_offers if offer[1:] != (_previous_offer['food'], _previous_offer['water'], _previous_offer['firewood'])]
            if _previous_offer and (best_offer == _previous_offer):
                logging.error(f"> !Best offer should not be the same as the previous offer. Best Offer: {best_offer} | Previous Offer: {_previous_offer}")
                #best_offer = max(final_scores, key=lambda x: x[1])[1]
                sorted_scores = sorted(final_scores, key=lambda x: x[1], reverse=True)
                # Select the second-highest value
                second_best_offer = sorted_scores[1][0]
                logging.info(f"> Change the best offer to the second best offer: {best_offer}")
        else: # Temp: NCR
            logging.info("> The top strategy is 'NCR' or 'RNC'. The BEST offer will be used.")
            logging.info("> Previous offer: %s", _previous_offer)
            logging.info("> Best offer: %s", best_offer)

        logging.info(">>> **Finally Selected Best Offer** : (%s: f=%s w=%s fw=%s)", best_offer[0], best_offer[1], best_offer[2], best_offer[3])
        logging.info("---")

        return best_offer, top_stg

    def process_fg_osad_results(self, results: List[Dict]):
        """Process assessment results to calculate assessment scores and top strategies."""
        #priority_alignment_list, total_score_list, fairness_list, coherence_list = [], [], [], []
        total_score_list, stg_intermediary_list = [], []
        scale= 10
        for result in results:

            #logging.debug(">> Assessment response: %s", assessment_response)
            if not isinstance(result, dict):
                logging.error(f"Error processing the assessment results: {result}")
                continue

            try:
                assessment_response = result["choices"]
                scores = [json.loads(x["message"]['content']) for x in assessment_response if x is not None]

                total_score = sum([x["TS"] for x in scores if "TS" in x]) / len(scores) / scale
                stg_intermediary = sum([x["SIIO"] for x in scores if "SIIO" in x]) / len(scores) / scale

                total_score_list.append(total_score)
                stg_intermediary_list.append(stg_intermediary)

            except Exception as e:
                logging.error(f"Error calculating the average score: {e}")
                logging.error(f"result: {result}")
                logging.error(f"assessment_response: {assessment_response}")
                logging.error(f"Scores: {scores}")
                raise

        return total_score_list, stg_intermediary_list

    def process_assessment_results(self, results: List[Dict]):
        """Process assessment results to calculate assessment scores and top strategies."""
        assessment_scores, top_strategies, strategy_dists = [], [], []
        stg_list = []
        for result in results:

            #logging.debug(">> Assessment response: %s", assessment_response)
            if not isinstance(result, dict):
                logging.error(f"Error processing the assessment results: {result}")
                continue

            try:
                assessment_response = result["choices"]
                scores = [json.loads(x["message"]['content']).get('score') for x in assessment_response if x is not None]
                validated_score = [x for x in scores if isinstance(x, (int, float))]
                assert len(validated_score) >= 3, f"# of Validated scores should be at least 3. rate of validated_score: {len(validated_score)/len(scores)}"
                avg_score = np.mean(validated_score)
            except Exception as e:
                logging.error(f"Error calculating the average score: {e}")
                logging.error(f"result: {result}")
                logging.error(f"assessment_response: {assessment_response}")
                logging.error(f"Scores: {scores}")
                raise

            for x in assessment_response:
                if x is not None:
                    strategy = json.loads(x["message"]['content']).get('strategy', '')

                    # Check if the strategy is a string and needs to be split or directly use it if it's a list
                    if isinstance(strategy, str):
                        strategy = strategy.split(",")
                    elif not isinstance(strategy, list):
                        strategy = []
                    stg_list.extend(strategy)

            normalized_scores = avg_score / 10 # normalize the score to 0-1. 5 is the maximum score
            strategy_dist = Counter(stg_list)
            strategy_dists.append(strategy_dist)
            assessment_scores.append(normalized_scores)
            #top_strategies.append(strategy_dist.most_common(1)[0][0])
            top_strategies.append(",".join([ f"{x[0]}({x[1]})" for x in strategy_dist.most_common(2)]))  # top 2 strategies

            # get top and second top strategies

        return assessment_scores, top_strategies, strategy_dists


    def process_fg_sa_results3(self, results: List[Dict], number_of_offers=5):
        """Process assessment results to calculate assessment scores and top strategies."""
        assessment_scores, top_strategies, strategy_dists = [], [], []
        stg_list = ["LIC", "CSC", "RC", "LGR", "FC", "MGF", "REO", "AIO", "AEO", "NCR", "RNC"]

        for result in results:
            if not isinstance(result, dict):
                logging.error(f"Error processing the assessment results: {result}")
                continue

            try:
                assessment_response = result["choices"]
                score_sets = []

                stgs= []
                offers = []
                rationales = []
                rank_dict = dict()
                for res in assessment_response:
                    stg= json.loads(res["message"]['content']).get("strategy")
                    offer = json.loads(res["message"]['content']).get("offer")
                    rationale = json.loads(res["message"]['content']).get("rationale")
                    ranks = json.loads(res["message"]['content']).get("rank")


                    if isinstance(stg, list):
                        stg= stg[0]

                    if stg in stg_list:
                        stgs.append(stg)
                        rationales.append(f"[{stg}] {rationale}")

                    if offer is None:
                        continue

                    if isinstance(ranks, list):
                        if contains_all_elements(ranks, number_of_offers):
                            if stg not in rank_dict:
                                rank_dict[stg]= []
                            rank_dict[stg].append(ranks)

                    score, f, w, fw = offer.get("score"), offer.get("food"), offer.get("water"), offer.get("firewood")

                    # check all the values are int
                    if not all(isinstance(x, int) for x in [score, f, w, fw]):
                        logging.error(f"Error processing the assessment results: {result}")
                        continue

                    #validation the score
                    if calculate_score({"food": f, "water": w, "firewood": fw}, self.agent_value_off_table) == score:
                        offers.append((score, f, w, fw))

                # get counter of offer and strategy
                strategy_dist = Counter(stgs)
                # get top 2 stg

                top_2_stgs = strategy_dist.most_common(2)
                top_stg = top_2_stgs[0][0]
                #top_stg= strategy_dist.most_common(1)[0][0]

                # get top 1 offers
                offer_dist = Counter(offers)
                top_offer = offer_dist.most_common(1)[0][0]

                print("Rank dict: ", rank_dict)
                print("strategy_dist: ", strategy_dist)
                rank_list = rank_dict[top_stg] # get top stg
                rank_sum = [sum(x) for x in zip(*rank_list)]

                # 리스트를 정렬하여 요소의 순위를 계산
                sorted_list = sorted(rank_sum, reverse=False)  # 내림차순으로 정렬
                final_ranks = [sorted_list.index(x) for x in rank_sum]  # 랭킹 부여


            except Exception as e:
                logging.error(f"Error : {e}")
                raise

        return top_2_stgs, top_offer, rationales, final_ranks


    def assess_offer(self, offer, offer_candidates:str, number_of_assessment=5, only_return_msg=False):
        """Assess the offer based on the negotiation strategy"""
        logging.debug("> Assess Offer")
        suggested_offer=None
        round_information = f"{self.cnt_agent_utterance(self.utterance_offer_history)+1} round / {self.args.n_round} rounds"
        if offer is not None:
            score = calculate_score(offer, self.agent_value_off_table)
            suggested_offer = f"[Score {score}] Food:{offer['food']}, Water:{offer['water']}, Firewood:{offer['firewood']}"

        proc_dialog_history=self.processed_dialog_history()
        proc_offer_history = self.process_utter_offer_history(self.utterance_offer_history, self.int_partner_priority, w_strategy=True, w_utterance=False, set_turn_index=True, role_user="PARTNER OFFER", role_assistant="YOUR OFFER")
        proc_concession_history = self.process_utter_offer_history(self.utterance_offer_history, self.int_partner_priority, w_utterance=False, set_turn_index=True, filter_None_offer=True, role='user')

        previous_STR_result = self.stg_first_step_results[-1] if self.stg_first_step_results else ''
        if previous_STR_result:
            lp_rationale, lp_max, lp_lambda, lp_fairness, lp_stance = previous_STR_result['rationale'], previous_STR_result['max_bound'], previous_STR_result['lambda'], previous_STR_result['partner_fairness'], previous_STR_result['partner_stance']
            previous_STR_result_str = f"Partner Offer Fairness={lp_fairness}\nPartner Stance={lp_stance}\nLP Max bound={lp_max}\nLP Lambda={lp_lambda}\nRationale={lp_rationale}"
        #logging.info("Generated LP parameters: %s", json.dumps(response, indent=4))


        round_information = f"{self.cnt_agent_utterance(self.utterance_offer_history)+1} round / {self.args.n_round} rounds"
        proc_dialog_history = self.processed_dialog_history()
        proc_offer_history = self.process_utter_offer_history(self.utterance_offer_history, self.int_partner_priority, w_strategy=True, w_utterance=False, role_user="PARTNER OFFER", role_assistant="YOUR OFFER")
        proc_concession_history = self.process_utter_offer_history(self.utterance_offer_history, self.int_partner_priority, w_utterance=False, set_turn_index=True, filter_None_offer=True, role='user')


        prompt=prompt_builder(self.agent_value_off_table, None, None, proc_dialog_history, suggested_offer=suggested_offer, offer_candidates=offer_candidates, offer_history=proc_offer_history, concession_history=proc_concession_history, prev_STR_results=previous_STR_result_str, expert_persona='all', prompt_type='fine_grained_self_assessment', verbose=False)
        msg= {"messages": [{ "role": "user", "content": prompt}], "n": number_of_assessment, "json_parsing_check": True, "model": self.engine_STR}
        if only_return_msg:
            return msg

        assesment_response=self.call_engine(**msg)
        assement_scores, top_strategies, strategy_dist = self.process_assessment_results([{"choices": assesment_response}])

        logging.debug(">> Offer(%s) is assessed with the average score: %s, top strategy: %s", suggested_offer, assement_scores[0], top_strategies[0])
        return {"score": assement_scores[0], "top_strategy": top_strategies[0]}


    def ask_for_priority_confirmation(self, **kwargs):
        """Generate a question for priority confirmation"""

        logging.debug("> [Process] Generate Question for Priority Confirmation")

        ask_for_needs_offer = kwargs.get("ask_for_need_offer", False)
        prompt_type = 'generate_questions_for_need_offer' if ask_for_needs_offer else 'generate_questions_for_priority'
        # Generate a question for priority confirmation
        inconcsistency_case = "NO" if self.is_partner_priority_consistent else "YES"
        proc_dialog_history = self.processed_dialog_history()
        generate_question_q_output = self.output_unconfirmed_items(self.priority_confirmation, self.item_priorities_remaining)
        generate_question_q=prompt_builder(self.agent_value_off_table, self.partner_priority, generate_question_q_output, proc_dialog_history, partner_prior_inconsistency=inconcsistency_case, prompt_type=prompt_type, verbose=self.verbose)
        #logging.debug("generate_question_q :\n%s", generate_question_q))
        #print("generate_question_q: ", generate_question_q)


        generated_question=self.call_engine(messages=[{ "role": "user", "content": generate_question_q}])
        logging.debug(">> Generated question for priority confirmation : %s", generated_question)
        final_question = generated_question['content'].replace("\n", " ")
        final_question = final_question.replace("\t", " ")
        # if not self.is_partner_priority_consistent:
        #     final_question = "It seems that the priorities you provided are inconsistent. " + final_question
        #     final_question = "It seems that the priorities you provided are inconsistent. " + final_question
        return final_question

    def check_priority_confirmation_and_updates(self):
        """
        check priority confirmation and update the priority confirmation
        """
        logging.debug("> [Process] Check Priority Confirmation and Update")
        if len(self.dialog_history) <= 2:
            return

        # get two latest dialog history
        recent_dialog_history = self.dialog_history[-2:]
        assert recent_dialog_history[-1]['role'] == 'user'

        proc_dialog_history = self.processed_dialog_history(user='ANSWER', assistant='QUESTION', num_of_latest_turns=2)
        priority_confirmation_q = prompt_builder(self.agent_value_off_table, self.partner_priority, self.priority_confirmation, proc_dialog_history, prompt_type='priority_confirmation', verbose=self.verbose)
        #logging.debug("priority_confirmation_q : %s", priority_confirmation_q)
        priority_confirmation_response=self.call_engine(messages=[{ "role": "user", "content": priority_confirmation_q}], json_parsing_check=True)
        #logging.debug(">> priority_confirmation_response : %s", priority_confirmation_response)

        priority_confirmation_response = json.loads(priority_confirmation_response["content"])
        confirmed_items = priority_confirmation_response.get("confirmed_items")

        if not confirmed_items:
            logging.info(">> No Key of 'confirmed_items'. Skipped to check the confirmed items this turn")
            return
        logging.debug(">> confirmed_items: %s", confirmed_items)
        logging.info(">> Confirmed_items (PC): %s", confirmed_items)

        # updates priorities if there is no same priority for two or more items in the response
        if self.check_duplicate_item_priorities(priority_confirmation_response) == False:
            for item in confirmed_items:
                confirmed_item = item["confirmed_item"]
                confirmed_priority = item["priority"]
                confirmed_meaning = item["meaning"]

                # update the priority confirmation
                self.update_priority_confirmation(confirmed_item, confirmed_priority, confirmed_meaning)

                # update item_priorities_remaining and fill in item/priorities when able
                self.item_priorities_remaining = self.update_item_priorities_remaining(confirmed_item, confirmed_priority, confirmed_meaning)

        #print(">> Item priorities remaining: ", self.item_priorities_remaining)
        logging.debug(" >> Item priorities remaining: %s", self.item_priorities_remaining)
        return

    def check_duplicate_item_priorities(self, priority_confirmation_response):
        output_dict = {'confirmed_items': []}

        # Loop through the priority_confirmation_response to create output_dict
        for i, item in enumerate(priority_confirmation_response['confirmed_items'], start=1):
            entry = {
                f'confirmed_item_{i}': item['confirmed_item'],
                f'priority_{i}': item['priority'],
                f'meaning_{i}': item['meaning']
            }
            output_dict['confirmed_items'].append(entry)

        priority_meaning_map = {}
        for item in output_dict['confirmed_items']:
            # Group keys by confirmed_item, priority, and meaning
            for key in item:
                if key.startswith('priority_'):
                    #index = key.augment('_')[1]
                    index = key.split('_')[1]
                    confirmed_item = item[f'confirmed_item_{index}']
                    priority = item[f'priority_{index}']
                    meaning = item[f'meaning_{index}']
                    if meaning == 'true':
                        if (priority, confirmed_item) in priority_meaning_map:
                            continue
                        elif priority in priority_meaning_map and priority_meaning_map[priority] != confirmed_item:
                            return True
                        else:
                            priority_meaning_map[priority] = confirmed_item
        return False

    def update_item_priorities_remaining(self, confirmed_item, confirmed_priority, confirmed_meaning):
        # sets all other priorities in item_priorities_remaining for a given item to 'false'
        def set_other_priorities_false(item, confirmed_priority):
            for priority in self.item_priorities_remaining[item]:
                if priority != confirmed_priority:
                    self.item_priorities_remaining[item][priority] = 'false'

        # sets the confirmed priority to 'false' for all other items in item_priorities_remaining
        def set_priority_false_for_other_items(confirmed_priority):
            for item in self.item_priorities_remaining:
                if item != confirmed_item:
                    self.item_priorities_remaining[item][confirmed_priority] = 'false'

        # checks item_priorities_remaining if only one priority is 'true' for an item, if so it updates other items accordingly
        def update_remaining_priority(item):
            num_false = sum(1 for status in self.item_priorities_remaining[item].values() if status == 'false')
            remaining_priority = next((priority for priority, status in self.item_priorities_remaining[item].items() if status == 'true'), 'null')
            if num_false == 2 and remaining_priority != 'null':
                logging.debug(">> Updating item priority using remaining priorities: %s, %s", item, remaining_priority)
                self.update_item_priorities_remaining(item, remaining_priority, 'true')

        # checks if two items are false for the same priority and updates the remaining item if necessary
        def confirm_third_item_for_priority(confirmed_priority):
            false_items = [item for item, priorities in self.item_priorities_remaining.items() if priorities.get(confirmed_priority) == 'false']
            if len(false_items) == 2:
                remaining_item = next(item for item in self.item_priorities_remaining if item not in false_items)
                if self.priority_confirmation[remaining_item] == False:
                    logging.debug(">> Confirming third item:", remaining_item, confirmed_priority)
                    self.update_item_priorities_remaining(remaining_item, confirmed_priority, 'true')

        if confirmed_meaning == 'true':
            # updates partner_priority and priority_confirmation
            self.partner_priority[confirmed_item] = confirmed_priority
            self.priority_confirmation[confirmed_item] = True

            # updates item_priorities_remaining with new confirmation
            set_other_priorities_false(confirmed_item, confirmed_priority)
            set_priority_false_for_other_items(confirmed_priority)

        elif confirmed_meaning == 'false':
            # sets the priority to 'false' in item_priorities_remaining
            self.item_priorities_remaining[confirmed_item][confirmed_priority] = 'false'

            # updates item_priorities_remaining if above line confirmed an item priority
            update_remaining_priority(confirmed_item)

            # checks if two items are false for the same priority, if so, updates third item
            confirm_third_item_for_priority(confirmed_priority)

        elif confirmed_meaning == 'null':
            return self.item_priorities_remaining

        # final confirmation if any more updates should occur, if so, updates accordingly
        for item in self.item_priorities_remaining:
            if not self.priority_confirmation[item]:
                update_remaining_priority(item)

        return self.item_priorities_remaining

    def update_priority_confirmation(self, confirmed_item, confirmed_priority, confirmed_meaning):
        """Update the priority confirmation based on the confirmed item and priority"""
        if confirmed_item == 'null' or confirmed_priority == 'null' or confirmed_meaning != 'true':
            logging.debug(">> Nothing to be updated; No priority confirmation is made")
            return

        # Check for conflicts between the priorities of items
        if any(item for item, priority in self.partner_priority.items() if priority == confirmed_priority and item != confirmed_item):
            conflict_item = [item for item, priority in self.partner_priority.items() if priority == confirmed_priority][0]
            logging.warning(f">> Confirmed_priority '{confirmed_priority}' is already assigned to item '{conflict_item}'; No priority confirmation is made")
            #raise ValueError(f"Confirmed_priority '{confirmed_priority}' is already assigned to item '{conflict_item}'")
            return
        # if self.partner_priority.get(confirmed_item) != 'null':
        #     logging.warning(f">> Confirmed_item '{confirmed_item}' is already assigned to priority '{self.partner_priority[confirmed_item]}'; No priority confirmation is made")
        #     raise ValueError(f"Confirmed_item '{confirmed_item}' is already assigned to priority '{self.partner_priority[confirmed_item]}'")

        # Update priority confirmation
        logging.debug("> Updating Priority Confirmation...")
        logging.debug(f"> Previous Priority Confirmation: {self.priority_confirmation}")
        self.priority_confirmation[confirmed_item] = True
        logging.debug(f">>> Updated Priority Confirmation: {self.priority_confirmation}")

        logging.debug("> Partner's priority updated")
        logging.debug(f"> Previous partner's priority: {self.partner_priority}")
        self.partner_priority[confirmed_item] = confirmed_priority
        logging.debug(f">>> Updated Partner's priority: {self.partner_priority}")

    def output_unconfirmed_items(self, priority_confirmation, item_priorities_remaining):
        output_list = []
        for item, confirmed in priority_confirmation.items():
            if not confirmed:
                remaining_priorities = [priority for priority, status in item_priorities_remaining[item].items() if status == 'true']
                remaining_priorities_str = ', '.join(remaining_priorities)
                output_list.append(f"The item, {item}, could be one of the following priorities: {remaining_priorities_str}")
        return '\n'.join(output_list)

    def process_utter_offer_history(self, utter_offer_history, int_partner_priority, w_utterance=True, w_strategy=False,  w_p_strategy=False, w_lp_params=False, w_offer=True, w_inferred_partner_priority=False, role_user="PARTNER", role_assistant="YOU", set_turn_index=False, filter_None_offer=False, num_of_latest_turns=None, role='all'):
        proc_utterances = []
        role_map = {"user": role_user, "assistant": role_assistant}
        #print(">> Utter Offer History: ", utter_offer_history)
        for idx, entry in enumerate(utter_offer_history):
            _role = role_map.get(entry['role'], entry['role'])
            _inferred_partner_priority = entry.get('inferred_partner_priority')
            #print("entry.get(['inferred_partner_priority']): ", entry.get('inferred_partner_priority'))
            inferred_partner_priority = convert_priority_str_to_int(_inferred_partner_priority) if _inferred_partner_priority and not check_null_value(_inferred_partner_priority) else ""
            content = entry['content']+ " " if entry['content'] and w_utterance else ""
            gen_params = entry.get('gen_params')
            if gen_params:
                LP_lambda = gen_params.get('lambda')
                max_bound = gen_params.get('max_bound')
                partner_offer_fairness = gen_params.get('partner_fairness')
                partner_stance = gen_params.get('partner_stance')

            offer = entry.get('offer')
            strategy = f"[Strategy: {entry['strategy']}] " if entry['role'] == 'assistant' and w_strategy else ""
            #filteing role
            if role != 'all' and role != entry['role']:
                continue
            if w_offer and offer is None:
                _offer = f"({role_map['user']}: No Offer)" if role == 'user' else "(YOU: None | PARTNER: None)"
            elif not w_offer:
                _offer = ""
            else:
                # Process partner's offer from STR
                #print("entry: ", entry)
                #print("inferred_partner_priority: ", inferred_partner_priority)
                #partner_offer = convert_item_cnts_partner(offer, int_partner_priority)
                partner_offer = convert_item_cnts_partner(offer, inferred_partner_priority)
                partner_offer_spec = f"Score={partner_offer[0]}: food={partner_offer[1]}, water={partner_offer[2]}, firewood={partner_offer[3]}"

                # Process agent's offer
                agent_score = calculate_score(offer, self.agent_value_off_table)
                agent_offer_spec = f"Score={agent_score}: food={offer['food']}, water={offer['water']}, firewood={offer['firewood']}"
                inferred_partner_priority = "| IPP: " + f"food={inferred_partner_priority['food']}, water={inferred_partner_priority['water']}, firewood={inferred_partner_priority['firewood']}" if w_inferred_partner_priority else ""
                lp_params = f" | P.F={partner_offer_fairness}, P.S={partner_stance} => MAX={max_bound}, LM={LP_lambda}" if w_lp_params and gen_params else ""
                if w_p_strategy: lp_params += " | STRATEGY: " + strategy_full_nm_mapper(entry.get('strategy', "no-mapping"), inverse=True)
                _offer = f"(PARTNER {partner_offer_spec}) {inferred_partner_priority}{lp_params}" if role == 'user' else f"(YOUR {agent_offer_spec} | PARTNER {partner_offer_spec}) {inferred_partner_priority}{lp_params}"


            proc_utterances.append(f"{_role}: {strategy}{content} {_offer}")

        if num_of_latest_turns:
            #assert num_of_latest_turns <= len(
            #    dialogues), "The number of latest dialogues should be less than the length of dialogues"
            if num_of_latest_turns <= len(proc_utterances):
                proc_utterances = proc_utterances[-num_of_latest_turns:]
            else:
                logging.debug("Since the number of latest dialogues is greater than the length of dialogues. We will use all dialogues.")

        if set_turn_index:
            if filter_None_offer:
                filtered_offer = [f"[Turn {idx}] {utterance}" for idx, utterance in enumerate(proc_utterances, start=1) if "No Offer" not in utterance]
                return "\n".join(filtered_offer) if len(filtered_offer) > 0 else "No offer (concession) is made yet."
            return "\n".join([f"[Turn {idx}] {utterance}" for idx, utterance in enumerate(proc_utterances, start=1)])

        return "\n".join(proc_utterances)


    def offer_grounded_generation(self, selected_offer):
        """Generate a grounded response for the selected offer"""
        logging.debug("> [Process] Offer Grounded Generation")
        # generated response
        proc_dialog_history = self.processed_dialog_history()
        proc_concession_history = self.process_utter_offer_history(self.utterance_offer_history, self.int_partner_priority, w_utterance=False, set_turn_index=True, filter_None_offer=True, role='user')
        grounded_response_q=prompt_builder(self.agent_value_off_table, self.partner_priority, self.priority_confirmation, proc_dialog_history, selected_offer=selected_offer, concession_history=proc_concession_history, prompt_type='offer_grounded_generation', verbose=self.verbose)
        generated_response=self.call_engine(messages=[{ "role": "user", "content": grounded_response_q}], json_parsing_check=True)
        offer_grounded_response = json.loads(generated_response["content"])["response"]
        #logging.debug("Offer grounded response : %s", offer_grounded_response)

        return offer_grounded_response

    def update_partner_priority(self, new_priority: Dict):
        """Manually Update the partner's priority based on the dialog history"""
        #Compare the new priority with the old one
        #If the item is not in the new priority, then keep the old one
        #If the items already confirmed, then keep the old one

        updated_priority = dict()
        for key, value in self.partner_priority.items():
            if key in new_priority and key not in [k for k, v in self.priority_confirmation.items() if v == True]:
                updated_priority[key] = new_priority[key]
            else:
                updated_priority[key] = value
        return updated_priority

    def set_maximum_value_for_LP(self):
        """Set the maximum value for LP"""
        if len(self.offer_history)==0:
            return 36
        latest_offer = self.offer_history[-1]
        score_of_latest_offer = 0
        for item, value in self.agent_value_off_table.items():
            score_of_latest_offer += value * latest_offer[item]

        if all([v for k, v in self.priority_confirmation.items()]): # all items are confirmed
            max_value = score_of_latest_offer
        else:
            max_value = score_of_latest_offer + 2

        return int(max_value)

    def calucate_score_both(self, offer):
        if offer is None:
            return None, None
        agent_score = calculate_score(offer, self.agent_value_off_table)
        partner_score = calculate_score(convert_item_cnts_partner(offer, only_cnts=True), self.int_partner_priority)
        return agent_score, partner_score

    def cache_and_update_lp_results(self, lp_key, results):
        if results:
            self.lp_results[lp_key] = list(results)
            cache_results(paths.CACHED_LP_RESULTS, self.lp_results)

    def set_OSAD_agent(self, OSAD_agent):
        self.OSAD_agent = OSAD_agent
        return

    @property
    def list_confirmed_items(self):
        """List items that are confirmed by the partner"""
        return self.priority_confirmation

    @property
    def inferred_partner_priority(self):
        """Infer the partner's priority from the dialog history"""
        return self.partner_priority

    @property
    def previous_offer(self):
        return self.offer_history[-1]

    @property
    def int_partner_priority(self):
        return convert_priority_str_to_int(self.partner_priority)

    @property
    def score_previous_offer(self):
        """Score the previous offer"""
        score = 0
        for item, value in self.agent_value_off_table.items():
            score += value * self.previous_offer[item]
        return score

    def is_priority_confirmed(self):
        return all(self.priority_confirmation.values())

    def reset_priorities(self, keys=None):
        """
        Resets the parter priorities of items to their default values and reset priority confirmation of the items.

        This method can reset specific items' priorities if a dictionary of keys is provided.
        If no dictionary is provided, it resets all items to their default priorities.

        :param keys: dict, optional
            A dictionary containing the names of items to be reset and their current priorities.
            If None, all items will be reset to their default priorities.
        """
        keys = keys if keys else ['water', 'food', 'firewood']
        for key in keys:
            self.partner_priority[key] = 'null'
            self.priority_confirmation[key] = False

        logging.debug(f'Resetting priorities \n>> Current partner priority : {self.partner_priority}\n'
                     f'>> Current priority confirmation : {self.priority_confirmation}')


class PartnerAgent(DialogAgent):

    def __init__(self,
                 agent_value_off_table: Dict,
                 initial_dialog_history=None,
                 agent_type="partner",  # "partner", "OSAD_agent"
                 engine="gpt-4o",
                 personality="base",
                 other_prompting=None,
                 system_instruction=None,
                 verbose=False
                 ):
        """Initialize the partner agent"""
        super().__init__(initial_dialog_history=initial_dialog_history,
                         agent_type=agent_type,
                         engine=engine,
                         system_instruction=system_instruction
                         )

        self.initialize_agent(initial_dialog_history, system_instruction)
        logging.debug(f"Initializing {agent_type} with engine {self.engine}")
        self.agent_value_off_table = agent_value_off_table
        self.verbose = verbose
        self.system_instruction = system_instruction
        self.personality = personality
        self.other_prompting = other_prompting
        assert self.personality in ["base", "greedy", "fair"], "Personality should be one of the following: base, greedy, fair"
        return

    def respond_wo_prompt(self, input):
        return self.respond(input)

    def respond(self, input, a2RL=False):
        self.dialog_history.append({"role": "user", "content": input})
        proc_dialog_history=self.processed_dialog_history()

        # logic for personality prompts
        _prompting = f"_{self.other_prompting}" if self.other_prompting else ""
        prompt_type = f"{self.personality}_partner{_prompting}_agent" if not a2RL else f"{self.personality}_partner_agent_a2RL"
        prompt=prompt_builder(self.agent_value_off_table, None, None, proc_dialog_history, prompt_type=prompt_type, verbose=self.verbose)
        #logging.info(">>> ======== Prompt for Partner Agent=========\n%s", prompt)
        #logging.info("\n")
        response=self.call_engine(messages=[{ "role": "user", "content": prompt}], json_parsing_check=True)
        response = json.loads(response["content"])['response']
        self.dialog_history.append({"role": "assistant", "content": response})
        return response



    def one_step_ahead_decision(self, agent_value_off_table:Dict, suggested_offer:str, offer_candidates:str, dialogue:list, num_of_decisions=4, only_return_msg=False):
        """Make a decision one step ahead (This is for virtual partner agent)"""
        logging.debug("> OSAD-Agent's One Step Ahead Decision")

        # Making question
        proc_dialog_history=self.processed_dialog_history(dialogues=dialogue, role_change=True)
        decision_q=prompt_builder(agent_value_off_table, None, None, proc_dialog_history, suggested_offer=suggested_offer, offer_candidates=offer_candidates, prompt_type='one_step_ahead_decision', verbose=self.verbose)
        #logging.info(">> Prompt for OSAD: \n%s", decision_q)
        #logging.debug("generate_question_q : %s", decision_q)

        msg= {"messages": [{ "role": "user", "content": decision_q}], "n": num_of_decisions, "json_parsing_check": True, "model": self.engine}
        if only_return_msg:
            return msg

        generated_decision=self.call_engine(**msg)
        #logging.debug("One step ahead decision: %s", generated_decision)

        return generated_decision

    def fine_grained_osad(self, agent_value_off_table:Dict, suggested_offer:str, offer_for_partner:tuple, offer_candidates:str, dialogue, number_of_assessment=5, only_return_msg=False):
        """Fine-grained assessment of the offer"""
        logging.debug("> Fine-grained Assessment of the Offer")

        offer_for_partner_str = f"[Score {offer_for_partner[0]}] Food:{offer_for_partner[1]}, Water:{offer_for_partner[2]}, Firewood:{offer_for_partner[3]}"

        proc_dialog_history=self.processed_dialog_history(dialogues=dialogue, role_change=True)
        prompt=prompt_builder(agent_value_off_table, None, None, proc_dialog_history, suggested_offer=suggested_offer, suggested_offer_for_partner=offer_for_partner_str, offer_candidates=offer_candidates, prompt_type='fine_grained_osad', verbose=False)
        #print('!!!!!!!!!!!!!!OSAD Prompt', prompt)

        msg= {"messages": [{ "role": "user", "content": prompt}], "n": number_of_assessment, "json_parsing_check": True, "model": self.engine}
        if only_return_msg:
            return msg

        assesment_response=self.call_engine(**msg)

        return assesment_response

    def reset(self):
        self.reset_dialog()


class ModeratorAgent(DialogAgent):
    """NOTE: initial experiments shows that the moderator is much better at recognizing deal than not deal
    Do not know why but interesting
    """
    def __init__(self,
                 initial_dialog_history=None,
                 agent_type="moderator",
                 engine="gpt-4o",
                 system_instruction=None,
                 trace_n_history=-1,
                 verbose=False
                ):
        """Initialize the moderator agent"""
        super().__init__(initial_dialog_history=initial_dialog_history,
                         agent_type=agent_type,
                         engine=engine,
                         system_instruction=system_instruction
                         )

        self.trace_n_history = trace_n_history
        self.verbose = verbose

        self.initialize_agent(initial_dialog_history, system_instruction)
        logging.debug("Initializing moderator with engine %s" % self.engine)
        return

    def check_status(self, dialog_history, trace_n_history=None):
        """Check if the negotiation is done given dialogue history"""
        if trace_n_history is None:
            trace_n_history = self.trace_n_history

        if self.trace_n_history != -1:
            assert len(dialog_history) >= self.trace_n_history, "The length of dialog history should be greater than the trace_n_history"
            dialog_history = dialog_history[-self.trace_n_history:]

        proc_dialog_history = self.processed_dialog_history(dialogues=dialog_history, assistant="PLAYER1", user="PLAYER2")
        prompt = prompt_builder(None, None, None, proc_dialog_history, prompt_type='moderator', verbose=self.verbose)
        response=self.call_engine(messages=[{ "role": "user", "content": prompt}], json_parsing_check=True)
        status = json.loads(response["content"])['answer']

        #processed response
        if "accept-deal" in status.lower():
            final_status = "ACCEPT-DEAL"
        elif "walk-away" in status.lower():
            final_status = "WALK-AWAY"
        elif "on-going" in status.lower():
            final_status = "ON-GOING"
        else:
            raise ValueError("Unknown status: %s from origianl GPT response %s" % (status, response))

        return final_status

    def moderate_conversation(self):
        """Moderate the conversation"""
        logging.debug("> [Process] Moderate Conversation")
        moderate_text = "Moderator: After the next 2 rounds of conversation, the negotiation will reach the maximum round and come to an end. Please hurry up and conclude the negotiation. If you fail to reach an agreement by the end, neither of you will receive anything (i.e., a score of 0 for both)."
        return moderate_text

    def check_statusRL(self, dialog_history, trace_n_history=None):
        """Check if the negotiation is done given dialogue history"""
        if trace_n_history is None:
            trace_n_history = self.trace_n_history

        if self.trace_n_history != -1:
            assert len(dialog_history) >= self.trace_n_history, "The length of dialog history should be greater than the trace_n_history"
            dialog_history = dialog_history[-self.trace_n_history:]

        proc_dialog_history = self.processed_dialog_history(dialogues=dialog_history, assistant="PLAYER1", user="PLAYER2")
        prompt = prompt_builder(None, None, None, proc_dialog_history, prompt_type='moderator', verbose=self.verbose)
        response=self.call_engine(messages=[{ "role": "user", "content": prompt}], json_parsing_check=True)
        status = json.loads(response["content"])['answer']
        last_user_utterance = ""
        for line in dialog_history[::-1]:
            if line['role'] == 'user':
                last_user_utterance = line['content']
                break
        #processed response
        if "accept-deal" in status.lower() or "<selection>" in last_user_utterance:
            final_status = "ACCEPT-DEAL"
        elif "walk-away" in status.lower():
            final_status = "WALK-AWAY"
        elif "on-going" in status.lower():
            final_status = "ON-GOING"
        else:
            raise ValueError("Unknown status: %s from origianl GPT response %s" % (status, response))

        return final_status
    def reset(self):
        self.reset_dialog()
