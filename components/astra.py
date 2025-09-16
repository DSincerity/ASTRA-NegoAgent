"""
ASTRA (Adaptive Strategic Reasoning with Action for Negotiations) Module

This module contains the ASTRA negotiation strategy that can be used independently
of the main agent class. It implements a three-stage strategic reasoning process:

1. Fairness and Stance Prediction
2. Tool (Linear Programming)-integrated Offer Optimization: LP Parameter Generation and Offer Simulation
3. Best Offer Selection using Partner Acceptance Probability (PAP) and Strategy Assessment (SA)
"""

import json
import logging
import asyncio
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple

from prompt.prompt_build import prompt_builder
from utils import calculate_score, strategy_full_nm_mapper, convert_item_cnts_partner, contains_all_elements
from tools import calculateBestOfferFromLP
from async_lib_api import call_multiple_apis


class ASTRA:
    """
    ASTRA Strategic Reasoning Module

    Performs strategic negotiation reasoning through three stages:
    1. Partner fairness and stance prediction
    2. Linear programming parameter generation and offer simulation
    3. Best offer selection based on acceptance probability and strategy assessment
    """

    def __init__(self, lp_caching_on: bool = True):
        """Initialize ASTRA module with optional LP caching"""
        self.lp_caching_on = lp_caching_on
        self.lp_results = {}  # Cache for LP results

    def predict_partner_fairness_stance(self,
                                      agent_value_table: Dict[str, int],
                                      utterance_offer_history: List,
                                      int_partner_priority: Dict[str, int],
                                      processed_dialog_history: str,
                                      process_utter_offer_history_func: callable,
                                      call_engine_func: callable,
                                      engine_str: str,
                                      n_round: int) -> Dict[str, Any]:
        """
        STAGE 1: Predict partner's fairness level and negotiation stance

        Args:
            agent_value_table: Agent's item valuations
            utterance_offer_history: History of offers and utterances
            int_partner_priority: Inferred partner priorities
            processed_dialog_history: Processed dialog for prompts
            process_utter_offer_history_func: Function to process offer history
            call_engine_func: Function to call LLM engine
            engine_str: LLM engine identifier
            n_round: Total negotiation rounds

        Returns:
            Dictionary containing partner_fairness and partner_stance
        """
        # Calculate current round information
        round_information = f"{self._count_agent_utterances(utterance_offer_history)+1} round / {n_round} rounds"

        # Process offer and concession history
        proc_offer_history = process_utter_offer_history_func(
            utterance_offer_history, int_partner_priority,
            w_strategy=True, w_utterance=False,
            role_user="PARTNER OFFER", role_assistant="YOUR OFFER",
            num_of_latest_turns=1
        )
        proc_concession_history = process_utter_offer_history_func(
            utterance_offer_history, int_partner_priority,
            w_utterance=False, set_turn_index=True,
            filter_None_offer=True, role='user',
            num_of_latest_turns=2
        )

        # Build prompts for fairness and stance prediction
        prompt_fairness = prompt_builder(
            agent_value_table, None, None, processed_dialog_history,
            offer_history=proc_offer_history, expert_persona='all',
            prompt_type='generate_LP_param_fairness', verbose=False
        )
        prompt_stance = prompt_builder(
            agent_value_table, None, None, processed_dialog_history,
            concession_history=proc_concession_history, expert_persona='all',
            prompt_type='generate_LP_param_stance', verbose=False
        )

        # Call LLM for predictions
        msg_fairness = {
            "messages": [{"role": "user", "content": prompt_fairness}],
            "model": engine_str,
            "json_parsing_check": True
        }
        msg_stance = {
            "messages": [{"role": "user", "content": prompt_stance}],
            "model": engine_str,
            "json_parsing_check": True
        }

        fair_response = call_engine_func(**msg_fairness)
        stance_response = call_engine_func(**msg_stance)

        fair_response = json.loads(fair_response["content"])
        stance_response = json.loads(stance_response["content"])

        fairness = fair_response.get("partner_offer_fairness")
        stance = stance_response.get("partner_stance")

        return {
            "partner_fairness": fairness,
            "partner_stance": stance
        }

    def determine_LP_params(self,
                           partner_fairness: str,
                           partner_stance: str,
                           agent_value_table: Dict[str, int],
                           utterance_offer_history: List,
                           int_partner_priority: Dict[str, int],
                           offer_history: List,
                           processed_dialog_history: str,
                           process_utter_offer_history_func: callable,
                           call_engine_func: callable,
                           engine_str: str,
                           n_round: int) -> Dict[str, Any]:
        """
        STAGE 2a: Determine Linear Programming parameters based on partner analysis

        Args:
            partner_fairness: Predicted partner fairness level
            partner_stance: Predicted partner stance
            agent_value_table: Agent's item valuations
            utterance_offer_history: History of offers and utterances
            int_partner_priority: Inferred partner priorities
            offer_history: History of offers made
            processed_dialog_history: Processed dialog for prompts
            process_utter_offer_history_func: Function to process offer history
            call_engine_func: Function to call LLM engine
            engine_str: LLM engine identifier
            n_round: Total negotiation rounds

        Returns:
            Dictionary containing LP parameters (lambda and max_bound)
        """
        # Calculate current round information
        round_information = f"{self._count_agent_utterances(utterance_offer_history)+1} round / {n_round} rounds"

        # Process offer and concession history
        proc_offer_history = process_utter_offer_history_func(
            utterance_offer_history, int_partner_priority,
            w_strategy=True, w_utterance=False,
            role_user="PARTNER OFFER", role_assistant="YOUR OFFER"
        )
        proc_concession_history = process_utter_offer_history_func(
            utterance_offer_history, int_partner_priority,
            w_utterance=False, set_turn_index=True,
            filter_None_offer=True, role='user'
        )

        # Get latest offer information
        latest_offer = offer_history[-1] if offer_history else ""
        latest_offer_str = (
            f"Score={calculate_score(latest_offer, agent_value_table)}: "
            f"Food={latest_offer['food']}, Water={latest_offer['water']}, "
            f"Firewood={latest_offer['firewood']}"
        ) if latest_offer else "No latest offer suggested"

        # Combine fairness and stance for prompt
        partner_fairness_stance = {
            "partner_fairness": partner_fairness,
            "partner_stance": partner_stance
        }

        # Build prompt for LP parameter generation
        prompt = prompt_builder(
            agent_value_table, None, None, processed_dialog_history,
            offer_history=proc_offer_history,
            concession_history=proc_concession_history,
            expert_persona='all',
            prompt_type='generate_LP_param_max_lambda',
            round=round_information,
            latest_offer=latest_offer_str,
            partner_fairness_stance=partner_fairness_stance,
            verbose=False
        )

        # Call LLM for LP parameters
        msg = {
            "messages": [{"role": "user", "content": prompt}],
            "model": engine_str,
            "json_parsing_check": True
        }
        raw_response = call_engine_func(**msg)
        response = json.loads(raw_response["content"])

        return response

    def simulate_lp_offers(self,
                          lp_query: Dict[str, Any],
                          agent_value_table: Dict[str, int],
                          int_partner_priority: Dict[str, int],
                          offer_history: List,
                          set_maximum_value_func: callable,
                          top_n: int = 5) -> List:
        """
        STAGE 2b: Generate optimal offer candidates using Linear Programming

        Args:
            lp_query: LP parameters from determine_LP_params
            agent_value_table: Agent's item valuations
            int_partner_priority: Inferred partner priorities
            offer_history: History of offers made
            set_maximum_value_func: Function to set max LP value (fallback only)
            top_n: Number of top offers to return

        Returns:
            List of potential offers from LP simulation
        """
        lambda_LP = lp_query["lambda"]
        maximum_val_LP = lp_query["max_bound"]

        # Validate maximum value bound
        score_of_latest_offer = (
            sum(agent_value_table[item] * offer_history[-1][item]
                for item in agent_value_table)
            if offer_history else 36
        )

        if int(maximum_val_LP) > score_of_latest_offer:
            logging.info(
                f">> Issue with generated max bound {maximum_val_LP} "
                f"(> score of latest offer({score_of_latest_offer})). "
                f"Setting it by rule (score_of_latest_offer)"
            )
            maximum_val_LP = score_of_latest_offer

        if not (maximum_val_LP > 10 and maximum_val_LP <= 36) or maximum_val_LP is None:
            logging.info(f">> Issue with generated max bound {maximum_val_LP}. Setting it by rule")
            maximum_val_LP = set_maximum_value_func()

        logging.info(">> Maximum value for LP: %s", maximum_val_LP)

        # Run core LP simulation
        potential_offers = self._simulate_lp_offers_core(
            maximum_val_LP, lambda_LP, agent_value_table,
            int_partner_priority, offer_history, top_n
        )

        return potential_offers

    def _simulate_lp_offers_core(self,
                               maximum_val_LP: float,
                               lambda_LP: float,
                               agent_value_table: Dict[str, int],
                               int_partner_priority: Dict[str, int],
                               offer_history: List,
                               top_n: int) -> List:
        """
        Core LP simulation method with caching and filtering logic

        Args:
            maximum_val_LP: Maximum value bound for LP
            lambda_LP: Lambda parameter for LP
            agent_value_table: Agent's item valuations
            int_partner_priority: Inferred partner priorities
            offer_history: History of offers made
            top_n: Number of top offers to return

        Returns:
            List of top LP-generated offers
        """
        # Check LP cache
        lp_cache_key = (
            maximum_val_LP,
            tuple(agent_value_table.items()),
            tuple(int_partner_priority.items())
        )

        if self.lp_caching_on and lp_cache_key in self.lp_results:
            logging.info('>> LP results: found cached results')
            cached_results = self.lp_results[lp_cache_key]
            return sorted(cached_results, key=lambda x: x[0], reverse=True)[:min(top_n, len(cached_results))]

        logging.debug('>> LP results: no cached results')

        # Execute the LP function to generate potential offers
        try:
            logging.debug(">> Executing LP function with the parameters.: max_value(%s), partner_priority(%s)",
                         maximum_val_LP, int_partner_priority)
            offer_list = calculateBestOfferFromLP(maximum_val_LP, lambda_LP, agent_value_table, int_partner_priority)

            # Cache results if caching is enabled
            if self.lp_caching_on:
                self.lp_results[lp_cache_key] = offer_list

        except Exception as e:
            logging.error(f"Error executing LP function: {e}")
            logging.error(f">> the parameters used: maximum_val_LP:{maximum_val_LP} | agent_value_table:{agent_value_table} | int_partner_priority:{int_partner_priority}")
            return []

        # Check if the LP function returned any offers
        if not offer_list:
            logging.error(f"No offers from LP. Please check the parameters for the LP function. maximum_val_LP:{maximum_val_LP} | agent_value_table:{agent_value_table} | int_partner_priority:{int_partner_priority}")
            return []

        # Sort offers by their total score in descending order
        sorted_offers = sorted(offer_list, key=lambda x: x[0], reverse=True)

        # Apply filtering logic if previous offer exists
        previous_offer = offer_history[-1] if offer_history else None
        if previous_offer:
            # Filter out offers not beneficial to both players
            agent_score_prev = calculate_score(previous_offer, agent_value_table)
            partner_score_prev = convert_item_cnts_partner(previous_offer, int_partner_priority)[0]
            filtered_offers = [
                offer for offer in sorted_offers
                if offer[0] >= agent_score_prev or
                convert_item_cnts_partner(offer, int_partner_priority)[0] >= partner_score_prev
            ]
            sorted_offers = filtered_offers

        if not sorted_offers:
            logging.error("All sorted offers were filtered out. No offers to return.")
            return []

        # Print sorted offers for debugging
        logging.debug(">> Offer list from LP:")
        for offer in sorted_offers:
            logging.debug(f"[Total Score: {round(offer[0])}] Food:{int(offer[1])}, Water:{int(offer[2])}, Firewood:{int(offer[3])}")

        # Return the top n offers
        top_offers = sorted_offers[:min(top_n, len(sorted_offers))]
        return top_offers

    def _process_fg_osad_results(self, results: List[Dict]) -> Tuple[List[float], List[float]]:
        """Process fine-grained OSAD assessment results to calculate total scores and strategy potential."""
        total_score_list, stg_intermediary_list = [], []
        scale = 10

        for result in results:
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

    def _process_fg_sa_results3(self, results: List[Dict], agent_value_table: Dict[str, int],
                               number_of_offers: int = 5) -> Tuple[List[Tuple], Tuple, List[str], List[int]]:
        """Process fine-grained self-assessment results to calculate strategy assessment."""
        stg_list = ["LIC", "CSC", "RC", "LGR", "FC", "MGF", "REO", "AIO", "AEO", "NCR", "RNC"]

        for result in results:
            if not isinstance(result, dict):
                logging.error(f"Error processing the assessment results: {result}")
                continue

            try:
                assessment_response = result["choices"]
                stgs = []
                offers = []
                rationales = []
                rank_dict = dict()

                for res in assessment_response:
                    content = json.loads(res["message"]['content'])
                    stg = content.get("strategy")
                    offer = content.get("offer")
                    rationale = content.get("rationale")
                    ranks = content.get("rank")

                    if isinstance(stg, list):
                        stg = stg[0]
                    if stg in stg_list:
                        stgs.append(stg)
                        rationales.append(f"[{stg}] {rationale}")
                    if offer is None:
                        continue
                    if isinstance(ranks, list):
                        if contains_all_elements(ranks, number_of_offers):
                            if stg not in rank_dict:
                                rank_dict[stg] = []
                            rank_dict[stg].append(ranks)

                    score, f, w, fw = offer.get("score"), offer.get("food"), offer.get("water"), offer.get("firewood")
                    # Check all the values are int
                    if not all(isinstance(x, int) for x in [score, f, w, fw]):
                        logging.error(f"Error processing the assessment results: {result}")
                        continue
                    # Validate the score
                    if calculate_score({"food": f, "water": w, "firewood": fw}, agent_value_table) == score:
                        offers.append((score, f, w, fw))

                # Get counter of offer and strategy
                strategy_dist = Counter(stgs)
                # Get top 2 strategies
                top_2_stgs = strategy_dist.most_common(2)
                top_stg = top_2_stgs[0][0]
                # Get top 1 offers
                offer_dist = Counter(offers)
                top_offer = offer_dist.most_common(1)[0][0]

                logging.debug("Rank dict: %s", rank_dict)
                logging.debug("strategy_dist: %s", strategy_dist)

                rank_list = rank_dict[top_stg]  # get top strategy
                rank_sum = [sum(x) for x in zip(*rank_list)]
                # Sort list to calculate element rankings
                sorted_list = sorted(rank_sum, reverse=False)  # ascending order
                final_ranks = [sorted_list.index(x) for x in rank_sum]  # assign rankings

            except Exception as e:
                logging.error(f"Error : {e}")
                raise

        return top_2_stgs, top_offer, rationales, final_ranks

    def _assess_offer_internal(self, agent_value_table: Dict[str, int], offer: Optional[Dict] = None,
                              offer_candidates: str = "", number_of_assessment: int = 5,
                              processed_dialog_history: str = "", utterance_offer_history: List = [],
                              int_partner_priority: Dict[str, int] = {}, n_round: int = 10,
                              stg_first_step_results: List = [], process_utter_offer_history_func: callable = None,
                              call_engine_func: callable = None, engine_str: str = "",
                              cnt_agent_utterance_func: callable = None) -> Dict[str, Any]:
        """Internal assess offer method for ASTRA use."""
        logging.debug("> Assess Offer")

        suggested_offer = None
        if offer is not None:
            score = calculate_score(offer, agent_value_table)
            suggested_offer = f"[Score {score}] Food:{offer['food']}, Water:{offer['water']}, Firewood:{offer['firewood']}"

        round_information = f"{cnt_agent_utterance_func(utterance_offer_history)+1} round / {n_round} rounds"

        proc_offer_history = process_utter_offer_history_func(
            utterance_offer_history, int_partner_priority, w_strategy=True, w_utterance=False,
            set_turn_index=True, role_user="PARTNER OFFER", role_assistant="YOUR OFFER"
        )
        proc_concession_history = process_utter_offer_history_func(
            utterance_offer_history, int_partner_priority, w_utterance=False,
            set_turn_index=True, filter_None_offer=True, role='user'
        )

        previous_STR_result = stg_first_step_results[-1] if stg_first_step_results else ''
        previous_STR_result_str = ""
        if previous_STR_result:
            lp_rationale = previous_STR_result['rationale']
            lp_max = previous_STR_result['max_bound']
            lp_lambda = previous_STR_result['lambda']
            lp_fairness = previous_STR_result['partner_fairness']
            lp_stance = previous_STR_result['partner_stance']
            previous_STR_result_str = (f"Partner Offer Fairness={lp_fairness}\n"
                                     f"Partner Stance={lp_stance}\n"
                                     f"LP Max bound={lp_max}\n"
                                     f"LP Lambda={lp_lambda}\n"
                                     f"Rationale={lp_rationale}")

        prompt = prompt_builder(
            agent_value_table, None, None, processed_dialog_history,
            suggested_offer=suggested_offer, offer_candidates=offer_candidates,
            offer_history=proc_offer_history, concession_history=proc_concession_history,
            prev_STR_results=previous_STR_result_str, expert_persona='all',
            prompt_type='fine_grained_self_assessment', verbose=False
        )

        return {
            "messages": [{"role": "user", "content": prompt}],
            "n": number_of_assessment,
            "json_parsing_check": True,
            "model": engine_str
        }

    def _strategic_reasoning_third_stg(self, offers_from_LP: List, osad_agent, agent_value_table: Dict[str, int],
                                     int_partner_priority: Dict[str, int], n_osad_decision: int, dialog_history: List,
                                     n_self_assessment: int, utterance_offer_history: List,
                                     stg_first_step_results: List, str3_logs: List, w1: float, w2: float,
                                     offer_history: List, processed_dialog_history: str,
                                     process_utter_offer_history_func: callable, call_engine_func: callable,
                                     engine_str: str, cnt_agent_utterance_func: callable, n_round: int) -> Tuple[Optional[List], str]:
        """
        Strategic reasoning third stage: Minimize regret using Partner Acceptance Probability and Strategy Assessment
        """
        logging.debug("> [Process] Strategic Reasoning Third Stage: Minimize Regret")

        assert osad_agent is not None, "OSAD_agent should be provided for one step ahead decision"

        osad_agent.setup_value_off_table(int_partner_priority)  # Set up OSAD_agent's value offer table with partner's priority

        osad_msgs, sa_msgs, offer_fineg_assessment = [], [], []

        # Process offers and generate processed offer candidates string
        processed_offer_candidates = []
        for idx, offer in enumerate(offers_from_LP):
            partner_score_items = convert_item_cnts_partner(offer, int_partner_priority)
            processed_offer_candidates.append(
                f"[Offer-index {idx}] Score={partner_score_items[0]}"
            )
        processed_offer_candidates_str = "\n".join(processed_offer_candidates)

        # Evaluate each offer and generate PAP messages
        partner_scores_list = []
        processed_agent_offer_candidates = []
        for offer_idx, offer in enumerate(offers_from_LP):
            partner_score_items = convert_item_cnts_partner(offer, int_partner_priority)
            suggested_offer_for_partner = (
                f"[Score {partner_score_items[0]}] Food: {partner_score_items[1]}, "
                f"Water: {partner_score_items[2]}, Firewood: {partner_score_items[3]}"
            )
            partner_scores_list.append(partner_score_items[0])

            # Fine-grained assessment of the offer
            osad_msgs.append(
                osad_agent.fine_grained_osad(
                    int_partner_priority,
                    suggested_offer_for_partner,
                    offer,
                    processed_offer_candidates_str,
                    dialog_history,
                    number_of_assessment=n_osad_decision,
                    only_return_msg=True
                )
            )

            processed_agent_offer_candidates.append(
                f"[Offer Candidate {offer_idx}] Score={offer[0]}: Food={offer[1]}, Water={offer[2]}, Firewood={offer[3]}"
            )

        processed_agent_offer_candidates_str = "\n".join(processed_agent_offer_candidates)
        sa_msgs.append(self._assess_offer_internal(
            agent_value_table, offer_candidates=processed_agent_offer_candidates_str,
            number_of_assessment=n_self_assessment, processed_dialog_history=processed_dialog_history,
            utterance_offer_history=utterance_offer_history, int_partner_priority=int_partner_priority,
            n_round=n_round, stg_first_step_results=stg_first_step_results,
            process_utter_offer_history_func=process_utter_offer_history_func,
            call_engine_func=call_engine_func, engine_str=engine_str,
            cnt_agent_utterance_func=cnt_agent_utterance_func
        ))

        # Asynchronous call to the engine
        combined_api_calls = osad_msgs + sa_msgs
        loop = asyncio.get_event_loop()
        combined_results = loop.run_until_complete(call_multiple_apis(combined_api_calls))
        results1 = combined_results[:len(osad_msgs)]
        results2 = combined_results[len(osad_msgs):]

        assert all(isinstance(x, dict) for x in combined_results), f"API results (OSAD + Assessment) should be a list of dictionaries. combined_results: {combined_results}"
        if not all(isinstance(x, dict) for x in combined_results):
            error_case = [x for x in combined_results if not isinstance(x, dict)]
            logging.error(f"!API results (OSAD + Assessment) Error case: {error_case}")

        # Processing the results
        # OSAD Results
        total_score, stg_potential = self._process_fg_osad_results(results1)

        # SA results
        top_2_stgs, top_offer, rationales, offer_ranks = self._process_fg_sa_results3(
            results2, agent_value_table, number_of_offers=len(offers_from_LP)
        )

        top_stgs_str = ",".join([f"{x[0]}({x[1]})" for x in top_2_stgs])
        top_stg = top_2_stgs[0][0]

        if top_offer != offers_from_LP[offer_ranks.index(0)]:
            logging.error(f"Top offer from the offer_ranks should be the same as the top_offer {top_offer} | {offers_from_LP[offer_ranks.index(0)]}")

        ranked_offers = [value for rank, value in sorted(zip(offer_ranks, offers_from_LP))]

        # ranking normalization from 0 to 1
        min_rank, max_rank = min(offer_ranks), max(offer_ranks)
        normalized_ranks = [(max_rank - rank) / (max_rank - min_rank) for rank in offer_ranks]

        # Fine-grained OSAD Results
        fg_w = [0.7, 0.3]
        fg_scores = list(zip(offers_from_LP, total_score, stg_potential, normalized_ranks, partner_scores_list))
        final_scores = [(offer, round(fg_w[0]*ts + fg_w[1]*sp, 2), round(ts, 2), round(sp, 2), round(norm_rank, 2), round(PS, 3)) for offer, ts, sp, norm_rank, PS in fg_scores]
        final_scores = [(x[0], round(w1*x[1] + w2*x[4], 2), x[1], x[2], x[3], x[4], x[5]) for x in final_scores]

        final_score_dict = {"offer_candidates": {}, "best_offers": {}}

        for idx, (offer, final_score, weighted_fg_OSAD, ts, sp, norm_rank, PS) in enumerate(final_scores):
            logging.info(f">>> Offer (%s): (%s: f=%s w=%s fw=%s) | Final W. Score: %s | OSAD: %s (TS: %s / SP: %s) | SA-Norm_Rank: %s | Partner Score: %s",
                        idx, offer[0], offer[1], offer[2], offer[3], final_score, weighted_fg_OSAD, ts, sp, norm_rank, PS)
            final_score_dict["offer_candidates"][idx] = {
                "Offer": offer, "Final_Score": final_score, "PAP": weighted_fg_OSAD,
                "TS": ts, "SP": sp, "Norm_Rank": norm_rank, "PartScore": PS
            }

        logging.info("---")

        # print Rationales for STR3
        logging.debug(" Top Stg: %s | Top Offer: %s | Rationales: \n%s", top_stgs_str, top_offer,
                    "\n".join(f"{index}: {item}" for index, item in enumerate(rationales, 1) if top_stg in item))
        logging.info(" Top Stg: %s | Top Offer: %s", top_stgs_str, top_offer)

        # Select best offers
        best_offer, best_offer_Score = (max(final_scores, key=lambda x: x[1])[0], max(final_scores, key=lambda x: x[1])[1]) if final_scores else (None, None)
        best_offer_only_PAP, best_offer_only_PAP_Score = (max(final_scores, key=lambda x: x[2])[0], max(final_scores, key=lambda x: x[2])[1]) if final_scores else (None, None)
        best_offer_only_SA, best_offer_only_SA_Score = (max(final_scores, key=lambda x: x[5])[0], max(final_scores, key=lambda x: x[5])[1]) if final_scores else (None, None)

        final_score_dict["best_offers"]["best_offer"] = {"Offer": best_offer, "Final_Score": best_offer_Score}
        final_score_dict["best_offers"]["best_offer_only_PAP"] = {"Offer": best_offer_only_PAP, "Final_Score": best_offer_only_PAP_Score}
        final_score_dict["best_offers"]["best_offer_only_SA"] = {"Offer": best_offer_only_SA, "Final_Score": best_offer_only_SA_Score}
        final_score_dict["stg_rationales"] = {index: item for index, item in enumerate(rationales) if top_stg in item}

        # Store logs
        str3_logs.append(final_score_dict)

        _previous_offer = offer_history[-1] if offer_history else None
        if top_stg not in ["NCR", "RNC"]:
            # Filter out the offer with the same offer in the previous offer
            if _previous_offer and (best_offer == _previous_offer):
                logging.error(f"> !Best offer should not be the same as the previous offer. Best Offer: {best_offer} | Previous Offer: {_previous_offer}")
                sorted_scores = sorted(final_scores, key=lambda x: x[1], reverse=True)
                # Select the second-highest value
                second_best_offer = sorted_scores[1][0]
                logging.info(f"> Change the best offer to the second best offer: {second_best_offer}")
                best_offer = second_best_offer
        else:  # Temp: NCR
            logging.info("> The top strategy is 'NCR' or 'RNC'. The BEST offer will be used.")
            logging.info("> Previous offer: %s", _previous_offer)
            logging.info("> Best offer: %s", best_offer)

        logging.info(">>> **Finally Selected Best Offer** : (%s: f=%s w=%s fw=%s)", best_offer[0], best_offer[1], best_offer[2], best_offer[3])
        logging.info("---")

        return best_offer, top_stg

    def select_best_offer(self,
                         potential_offers: List,
                         # All required parameters for strategic reasoning
                         osad_agent,
                         agent_value_table: Dict[str, int],
                         int_partner_priority: Dict[str, int],
                         n_osad_decision: int,
                         dialog_history: List,
                         n_self_assessment: int,
                         utterance_offer_history: List,
                         stg_first_step_results: List,
                         str3_logs: List,
                         w1: float,
                         w2: float,
                         offer_history: List,
                         processed_dialog_history: str,
                         process_utter_offer_history_func: callable,
                         call_engine_func: callable,
                         engine_str: str,
                         cnt_agent_utterance_func: callable,
                         n_round: int) -> Tuple[Optional[List], str]:
        """
        STAGE 3: Select the best offer using Partner Acceptance Probability and Strategy Assessment

        This is essentially a wrapper around strategic_reasoning_third_stg, matching the original design.
        """
        assert potential_offers, "No offers from LP. Please check the parameters for the LP function."

        # Call strategic reasoning third stage (this is the main logic)
        selected_offer, selected_strategy = self._strategic_reasoning_third_stg(
            potential_offers, osad_agent, agent_value_table, int_partner_priority,
            n_osad_decision, dialog_history, n_self_assessment, utterance_offer_history,
            stg_first_step_results, str3_logs, w1, w2, offer_history,
            processed_dialog_history, process_utter_offer_history_func,
            call_engine_func, engine_str, cnt_agent_utterance_func, n_round
        )

        # Update offer history with selected or fallback offer (matching original logic)
        if selected_offer:
            offer_details = dict(zip(['food', 'water', 'firewood'], selected_offer[1:]))
            strategy_name = strategy_full_nm_mapper(selected_strategy)
            logging.debug(">> Finally Selected Best Offer: [Score %s] %s", selected_offer[0], offer_details)
        else:
            logging.debug(">> No suitable offer provided from LP. Using the previous offer.")
            offer_details = offer_history[-1] if offer_history else None  # Fallback to the previous offer if no suitable one found
            strategy_name = "None"

        return selected_offer, strategy_name

    def run_astra_pipeline(self,
                          # Required data
                          agent_value_table: Dict[str, int],
                          utterance_offer_history: List,
                          int_partner_priority: Dict[str, int],
                          offer_history: List,

                          # Required functions from agent
                          processed_dialog_history: str,
                          process_utter_offer_history_func: callable,
                          call_engine_func: callable,
                          set_maximum_value_func: callable,

                          # Configuration
                          engine_str: str,
                          n_round: int,
                          top_n: int = 5,

                          # Strategic reasoning parameters (for complete internal operation)
                          osad_agent = None,
                          dialog_history: List = None,
                          n_osad_decision: int = 5,
                          n_self_assessment: int = 5,
                          stg_first_step_results: List = None,
                          str3_logs: List = None,
                          w1: float = 0.35,
                          w2: float = 0.65,

                          # Optional storage lists
                          generated_params_list: Optional[List] = None) -> Tuple[Optional[List], str]:
        """
        Complete ASTRA pipeline execution

        Args:
            agent_value_table: Agent's item valuations
            utterance_offer_history: History of offers and utterances
            int_partner_priority: Inferred partner priorities
            offer_history: History of offers made
            processed_dialog_history: Processed dialog for prompts
            process_utter_offer_history_func: Function to process offer history
            call_engine_func: Function to call LLM engine
            lp_simulation_func: Function to run LP simulation
            strategic_reasoning_func: Function to perform strategic reasoning
            set_maximum_value_func: Function to set max LP value
            engine_str: LLM engine identifier
            n_round: Total negotiation rounds
            top_n: Number of top offers to return
            generated_params_list: Optional list to store generated LP params
            selected_strategy_list: Optional list to store selected strategies

        Returns:
            Tuple of (selected_offer, selected_strategy_name)
        """
        # STAGE 1: Fairness and Stance Prediction
        partner_fairness_stance = self.predict_partner_fairness_stance(
            agent_value_table=agent_value_table,
            utterance_offer_history=utterance_offer_history,
            int_partner_priority=int_partner_priority,
            processed_dialog_history=processed_dialog_history,
            process_utter_offer_history_func=process_utter_offer_history_func,
            call_engine_func=call_engine_func,
            engine_str=engine_str,
            n_round=n_round
        )

        partner_fairness = partner_fairness_stance["partner_fairness"]
        partner_stance = partner_fairness_stance["partner_stance"]

        # STAGE 2a: Generate LP parameters
        lp_query = self.determine_LP_params(
            partner_fairness=partner_fairness,
            partner_stance=partner_stance,
            agent_value_table=agent_value_table,
            utterance_offer_history=utterance_offer_history,
            int_partner_priority=int_partner_priority,
            offer_history=offer_history,
            processed_dialog_history=processed_dialog_history,
            process_utter_offer_history_func=process_utter_offer_history_func,
            call_engine_func=call_engine_func,
            engine_str=engine_str,
            n_round=n_round
        )

        # Store generated parameters if list provided
        if generated_params_list is not None:
            generated_params_list.append(lp_query)

        # STAGE 2b: Execute LP simulation
        potential_offers = self.simulate_lp_offers(
            lp_query=lp_query,
            agent_value_table=agent_value_table,
            int_partner_priority=int_partner_priority,
            offer_history=offer_history,
            set_maximum_value_func=set_maximum_value_func,
            top_n=top_n
        )

        # STAGE 3: Select best offer
        assert osad_agent is not None, "OSAD_agent should be provided for one step ahead decision"
        #if osad_agent is not None and dialog_history is not None:
            # Use internal strategic reasoning (complete ASTRA)
        selected_offer, strategy_name = self.select_best_offer(
            potential_offers=potential_offers,
            osad_agent=osad_agent,
            agent_value_table=agent_value_table,
            int_partner_priority=int_partner_priority,
            n_osad_decision=n_osad_decision,
            dialog_history=dialog_history,
            n_self_assessment=n_self_assessment,
            utterance_offer_history=utterance_offer_history,
            stg_first_step_results=stg_first_step_results or [],
            str3_logs=str3_logs or [],
            w1=w1,
            w2=w2,
            offer_history=offer_history,
            processed_dialog_history=processed_dialog_history,
            process_utter_offer_history_func=process_utter_offer_history_func,
            call_engine_func=call_engine_func,
            engine_str=engine_str,
            cnt_agent_utterance_func=self._count_agent_utterances,
            n_round=n_round
        )

        return selected_offer, strategy_name

    def _count_agent_utterances(self, utterance_offer_history: List) -> int:
        """Helper method to count agent utterances in history"""
        if not utterance_offer_history:
            return 0
        return sum(1 for item in utterance_offer_history if item.get('role') == 'assistant')
