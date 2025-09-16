import json
import logging
import os
import paths
from copy import deepcopy
from typing import Dict, Optional
from collections import Counter

from .base_dialog_agent import DialogAgent
from components import (
    PriorityConsistencyChecker,
    ASTRA,
    PartnerPreferenceAsker,
    PartnerPreferenceUpdater
)
from prompt.prompt_build import prompt_builder
from utils import check_null_value, calculate_score, convert_item_cnts_partner, lower_key_dict, convert_priority_str_to_int, cache_results, get_cached_results, strategy_full_nm_mapper, sync_partner_priorirty_confimation, set_inital_partner_priority, validate_offer

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
        # Initialize ASTRA strategic reasoning module
        self.astra = ASTRA(lp_caching_on=True)
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
        self.priority_consistency_checker = PriorityConsistencyChecker(
            agent=self,
            system_instruction=system_instruction,
            verbose=self.inconsistency_verbose
        )
        self.preference_asker = PartnerPreferenceAsker(
            prompt_builder=prompt_builder,
            call_engine=self.call_engine,
            logger=logging,
            verbose=self.verbose,
        )
        self.preference_updater = PartnerPreferenceUpdater(agent=self)

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
        action, response, agent_offer, _ = response_mode
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
            self.preference_updater.check_priority_confirmation_and_updates()
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
                logging.info(f">> Stopped asking the partner's priority. Already asked twice. Asking needs or offers, asking cnt:{self.asking_priority_cnt}")
                #self.priority_asker_on = False
                self.asking_priority_cnt = 0
                # for the case where the partner's priority is inconsistent, don't set the partner priroriy to the opposite one
                #if self.is_partner_priority_consistent is True:
                if not self.inconsistency_detected_case: # This variable is updated in the consistency checker
                    self.partner_priority= set_inital_partner_priority(self.partner_priority, self.priority_confirmation, self.agent_value_off_table)
                    logging.debug("> Set partner priroity to the opposite one: %s", self.partner_priority)
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
            logging.info(">> Inconsistency for the confirmed items. Turned on the priority checker and asker")
            self.priority_checker_on = True
            self.priority_asker_on = True
            self.asking_priority_cnt += 1
            self.is_counter_offer = False # counter offers는 없었던 것으로 간주.
            return ("ASKING", self.ask_for_priority_confirmation(), None, None)

        # Add the last offer to the offer history
        if self.is_counter_offer:
            logging.debug(">> Partner Counter offer is made. ")
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
        logging.debug("compromised_decision_before_end: %s | is_counter_offer: %s", self.compromised_decision_before_end, self.is_counter_offer)
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

        # Run ASTRA pipeline using the modular ASTRA class
        selected_offer, strategy_name = self.astra.run_astra_pipeline(
            # Required data
            agent_value_table=self.agent_value_off_table,
            utterance_offer_history=self.utterance_offer_history,
            int_partner_priority=self.int_partner_priority,
            offer_history=self.offer_history,

            # Required functions from agent
            processed_dialog_history=self.processed_dialog_history(),
            process_utter_offer_history_func=self.process_utter_offer_history,
            call_engine_func=self.call_engine,
            set_maximum_value_func=self.set_maximum_value_for_LP,

            # Configuration
            engine_str=self.engine_STR,
            n_round=self.args.n_round,
            top_n=self.args.top_n,

            # Strategic reasoning parameters
            osad_agent=self.OSAD_agent,
            dialog_history=self.dialog_history,
            n_osad_decision=self.n_OSAD_decision,
            n_self_assessment=self.n_self_assessment,
            stg_first_step_results=self.stg_first_step_results,
            str3_logs=self.STR3_logs,
            w1=self.w1,
            w2=self.w2,

            # Optional storage lists
            generated_params_list=self.generated_params
        )

        # Handle the result and update offer history
        if selected_offer:
            offer_details = dict(zip(['food', 'water', 'firewood'], selected_offer[1:]))
            self.offer_history.append(offer_details)
            self.selected_strategy.append(strategy_full_nm_mapper(strategy_name))
            logging.debug(">> Finally Selected Best Offer: [Score %s] %s" , selected_offer[0], offer_details)
        else:
            # Fallback to previous offer if no suitable offer found
            offer_details = self.previous_offer
            self.offer_history.append(offer_details)
            self.selected_strategy.append("None")

        # Generate a response grounded in the selected offer
        return self.generate_grounded_response(selected_offer, with_top_strategy=False)

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
        _priority_response = json.loads(priority_response["content"])

        # temporary unmark
        if not self.priority_asker_on or only_priority_prediction:  # Updating when not asking or only the predition module is on.
            logging.info(">> [STG-1] Priority Update Process....")
            inferred_partner_priority = lower_key_dict(_priority_response['Q1']["Answer"])
            partner_prioriry_to_be_updated = lower_key_dict(_priority_response['Q2']["Answer"])

            # Update only if it’s unconfirmed and the priority has a null value.
            if not self.is_priority_confirmed() and check_null_value(partner_priority):
                partner_prioriry_to_be_updated=self.update_partner_priority(partner_prioriry_to_be_updated)

                # update partner's priority
                if self.partner_priority != partner_prioriry_to_be_updated:
                    logging.info("\n%s\n[STG-1] Partner's priority will be updated from (Current) %s to (Updated) %s\n%s\n", "*"*50, self.partner_priority, partner_prioriry_to_be_updated, "*"*50)
                    self.partner_priority = partner_prioriry_to_be_updated
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

        # Validate the extracted offer
        if self.priority_consistency_checker_on:
            if self.is_counter_offer:
                # We can assert the consistency between the counter offer from the consistency checker and the one from STR. But here, we will change the partner_offer_history based on the consistency checker.
                #assert self.partner_offer_history[-1] == partner_offer, "The counter offer from the consistency checker is not consistent with the one from STR"

                if self.partner_offer_history[-1] != partner_offer:
                    logging.critical(
                        "[Partner Offer Inconsistency between Consistency Checker and STR] Check offers between the Consistency Checker (CC) and STR 1st stage..\n"
                        f"partner offer from CC : {self.partner_offer_history[-1]}\n"
                        f"partner offer from STR : {partner_offer}")

                    _dialog_history = self.processed_dialog_history(num_of_latest_turns=1)
                    double_checked_final_offer = self.double_check_partner_offer(n=1, dialog_history=_dialog_history)

                    # Correct partner offer history after double-checking
                    reextracted_partner_offer = convert_item_cnts_partner(double_checked_final_offer, only_cnts=True)

                    # Change the partner_offer to the one from the consistency checker into one from STR

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

    def ask_for_priority_confirmation(self, **kwargs):
        return self.preference_asker.ask(
            agent_value_off_table=self.agent_value_off_table,
            partner_priority=self.partner_priority,
            priority_confirmation=self.priority_confirmation,
            item_priorities_remaining=self.item_priorities_remaining,
            dialog_history_processed=self.processed_dialog_history(),
            is_partner_priority_consistent=self.is_partner_priority_consistent,
            ask_for_need_offer=kwargs.get("ask_for_need_offer", False),
        )



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
        _prompting = f"_{self.other_prompting}" if self.other_prompting != "base" else ""
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
