import logging

from prompt.prompt_build import prompt_builder
from agent.base_dialog_agent import DialogAgent
from utils import *



class PriorityConsistencyChecker(DialogAgent):
    def __init__(self, agent, system_instruction, verbose=False, engine="gpt-4o-mini"):
        super().__init__(engine=engine, system_instruction=system_instruction)
        self.agent = agent
        self.verbose = verbose

    def validate_items_offered(self, offer):
        """
        Validates and processes the items being offered.

        Validation ensures that the number of each item in the offer is within the valid range (0 to 3).
        If the offer type is "give", it adjusts the offer to reflect the items that will remain
        after giving the specified amounts.

        :param offer: dictionary that contains the characteristics of the offer and the items
        :return dictionary of items that will be obtained by the offerer
        """

        if offer == {}:
            return {}

        assert all(
            0 <= value <= 3 for key, value in offer.items()), "Offer contains item count outside the range 0-3."

        return offer

    def extract_negotiation_info(self, dialog_history):
        """
        Extracts agent"s offer and partner"s response from the previous two dialogs
        """
        consistency_check_prompt = prompt_builder(agent_value_off_table=None, partner_inferred_priority=None,
                                                  priority_confirmation=None,
                                                  conversation_history=dialog_history,
                                                  prompt_type="consistency_check",
                                                  verbose=self.verbose)
        message = self.call_engine(messages=[{"role": "user", "content": consistency_check_prompt}],
                                   json_parsing_check=True)

        _extracted_info = {}
        content = json.loads(message["content"])
        _extracted_info["gpt_contains_partner_offer"] = content["contains_partner_offer"] == "T"
        _partner_offer = self.validate_items_offered(content['partner_offer']['items_you_will_get'])
        _extracted_info['gpt_partner_offer'] = {key: 3 - value for key, value in _partner_offer.items()}

        _extracted_info["gpt_mentioned_item_preference"] = content["mentioned_item_preference"]

        if self.verbose:
            logging.debug(f"[ConsistencyChecker] Information extracted : \n{content})")

        return _extracted_info

    def _check_partner_offer_consistency(self, partner_priority, agent_offer, partner_offer):
        """
        Verifies the consistency of the counteroffer by comparing its value to the value of the original offer.

        :param partner_priority: dictionary that contains values of firewood, water and food
        :param agent_offer: dictionary containing the quantities of firewood, water, and food being offered.
        :param partner_offer: dictionary containing the quantities of firewood, water, and food being counteroffered.
        :return True if partner will get less value from the counteroffer than the original offer
        """
        if not agent_offer:
            # TODO(jiwon): Check whether the partner offer is aligned with the priority
            logging.debug(
                f"> [ConsistencyChecker] No agent offers... Check the number of item requested matches the partner priority")
            return True, True

        partner_value_with_agent_offer = sum(
            partner_priority[item] * (3 - int(agent_offer[item])) for item in agent_offer)
        partner_value_with_partner_offer = sum(
            partner_priority[item] * (3 - int(partner_offer[item])) for item in partner_offer)

        consistency_TF = partner_value_with_agent_offer <= partner_value_with_partner_offer
        confirmed_item_TF = self.agent.is_priority_confirmed()
        if not consistency_TF:
            logging.info(f"> [ConsistencyChecker] Checking priority consistency based on the partner's counter offer...")
            logging.info(
                    f">>> partner score in agent_offer : {partner_value_with_agent_offer} vs partner's score in partner offer : {partner_value_with_partner_offer}")
            logging.info(f"> partner priority:  {str(partner_priority)} | agent score in agent recent offer: {str(agent_offer)} | agent score in partner recent offer: {str(partner_offer)}")
        return consistency_TF, confirmed_item_TF

    @staticmethod
    def _json_key_check(json_obj, key, lower_case=True, default_value=None):
        if key in json_obj and json_obj[key] in ['food', 'water', 'firewood']:
            return json_obj[key].lower() if lower_case else json_obj[key]
        return default_value

    def _check_priority_order_consistency(self, partner_priority, extracted_priority):
        """
        Verifies the consistency of the rejection by comparing the item’s value to the extracted priority information

        :param partner_priority: dictionary that contains values of firewood, water and food
        :param extracted_priority: A dictionary containing the priority information extracted from the response

        :return: A tuple containing consistency of the most, the least preferred item, the consistency of the priority
                order of items and the list of items with inconsistent priorities
        :rtype: tuple(bool, bool, bool, list)
        """
        most_preferred_item = max(partner_priority, key=partner_priority.get)
        least_preferred_item = min(partner_priority, key=partner_priority.get)

        gpt_most_preferred_item = self._json_key_check(json_obj=extracted_priority, key='most_preferred_item')
        gpt_least_preferred_item = self._json_key_check(json_obj=extracted_priority, key='least_preferred_item')
        gpt_preference_order = extracted_priority[
            'order_of_preference'] if 'order_of_preference' in extracted_priority else None

        is_order_of_preference_consistent = True
        items_with_inconsistent_priority = set()

        if gpt_preference_order:
            ordered_item = [i.strip() for i in gpt_preference_order.split(">")]
            for index in range(len(ordered_item) - 1):
                item = ordered_item[index]
                next_item = ordered_item[index + 1]
                if item not in ['food', 'water', 'firewood'] or next_item not in ['food', 'water', 'firewood']:
                    continue

                partner_current_item_value = partner_priority[item]
                partner_next_item_value = partner_priority[next_item]

                is_order_of_preference_consistent = partner_current_item_value > partner_next_item_value
                if not is_order_of_preference_consistent: # Only check for the confirmed items
                    items_with_inconsistent_priority.update(ordered_item)
                    break

        is_most_preferred_item_consistent = most_preferred_item == gpt_most_preferred_item or not gpt_most_preferred_item
        is_least_preferred_item_consistent = least_preferred_item == gpt_least_preferred_item or not gpt_least_preferred_item

        if not is_most_preferred_item_consistent:
            items_with_inconsistent_priority.add(gpt_most_preferred_item)

        if not is_least_preferred_item_consistent:
            items_with_inconsistent_priority.add(gpt_least_preferred_item)

        confirmed_item_TF = all([self.agent.priority_confirmation[i] for i in  items_with_inconsistent_priority]) if items_with_inconsistent_priority else True
        consistency_TF = is_most_preferred_item_consistent and is_least_preferred_item_consistent and is_order_of_preference_consistent

        if not consistency_TF:
            logging.info("> [ConsistencyChecker] Checking priority consistency based on the priority ORDER in the partners response...)")
            logging.info(
                f">>> Extracted info from response: most preferred:{gpt_least_preferred_item} | least preferred:{gpt_least_preferred_item} | order of preference : {gpt_preference_order}")
            logging.info(f"items_with_inconsistent_priority: {list(items_with_inconsistent_priority)}")
        return consistency_TF, confirmed_item_TF

    def check_consistency(self, update_partner_priority):
        """
        Checks consistency in the priority of the partner given the offer and the priority information extracted from the dialog


        :param update_partner_priority: Boolean value to check whether to update partner priority based on the inconsistency result
        :return: A tuple containing whether the previous dialog was offer and response, consistency of agent"s item priority,
                 inconsistent item, existence of counteroffer, and items counteroffered
        :rtype: tuple(bool, bool, list, bool, dict)
        """
        dialog_history = self.agent.processed_dialog_history(num_of_latest_turns=1, assistant='YOU')
        partner_priority = self.agent.partner_priority

        if check_null_value(partner_priority):
            return

        result = self.extract_negotiation_info(dialog_history) # call API
        partner_priority = convert_priority_str_to_int(partner_priority)

        is_partner_counteroffer = result["gpt_contains_partner_offer"]
        self.agent.is_counter_offer = False  # Default value
        self.agent.is_partner_priority_consistent = True  # Default value
        self.agent.is_inconsistent_item_confirmed = True  # Default value

        if is_partner_counteroffer:
            # When partner offer is available : Infer priority from the items counteroffered
            logging.debug(f"> [ConsistencyChecker] Found partner offer... Analyzing inconsistency")
            self.agent.is_counter_offer = True

            gpt_items_partner = result["gpt_partner_offer"]
            self.agent.partner_offer_history.append(gpt_items_partner)

            self.agent.is_partner_priority_consistent, self.agent.is_inconsistent_item_confirmed = self._check_partner_offer_consistency(
                partner_priority=partner_priority,
                agent_offer=self.agent.offer_history[-1] if len(self.agent.offer_history) > 0 else None,
                partner_offer=gpt_items_partner)

            logging.debug(f"self.agent.is_partner_priority_consistent: {self.agent.is_partner_priority_consistent}")
            logging.info(
                f'[ConsistencyChecker] Offer is {"consistent" if self.agent.is_partner_priority_consistent else "inconsistent"} with partner priority based on the inconsistency in the counteroffer')

            if not update_partner_priority:
                logging.info('[ConsistencyChecker] Updating partner priority is off... Leaving consistency checker..')
                return

            if not self.agent.is_partner_priority_consistent:
                logging.info("[ConsistencyChecker] Resetting priorities...")
                self.agent.reset_priorities()
                self.agent.utterance_offer_history = []  # Reset all utterance offers
                self.agent.offer_history = []
                self.agent.inconsistency_detected_case = True
                self.agent.setup_decision_parameters()
                self.agent.partner_offer_history = []
                #self.agent.partner_offer_history = []
        else:
            # When partner didn't offer a counter offer : Infer priority from their utterance
            logging.info(f"[ConsistencyChecker] Contains no offer...")
            #self.agent.partner_offer_history.append({'food': None, 'water': None, 'firewood': None})
            self.agent.partner_offer_history.append(None)

            if not update_partner_priority:
                logging.debug('[ConsistencyChecker] Updating partner priority is off... Leaving consistency checker..')
                return

            self.agent.is_partner_priority_consistent, self.agent.is_inconsistent_item_confirmed = self._check_priority_order_consistency(
                partner_priority=partner_priority,
                extracted_priority=result["gpt_mentioned_item_preference"])

            logging.info(
                f'[ConsistencyChecker] Offer is {"consistent" if self.agent.is_partner_priority_consistent else "inconsistent"} with partner priority based on the relative priority inconsitency in the utterance')

            # Reset priority if partner priority is inconsistent
            if not self.agent.is_partner_priority_consistent:
                logging.info("> [ConsistencyChecker] Resetting priorities...")
                # self.agent.reset_priorities(inconsistent_items)
                self.agent.reset_priorities()  # Reset all priorities
                self.agent.utterance_offer_history = []  # Reset all utterance offers
                self.agent.partner_offer_history = []
                self.agent.offer_history = []
                self.agent.inconsistency_detected_case = True
                self.agent.setup_decision_parameters()
                #self.agent.partner_offer_history = []  # Reset all partner offers
