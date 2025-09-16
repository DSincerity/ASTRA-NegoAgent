import json
import logging
from typing import Dict, Any, List, Callable


class PartnerPreferenceUpdater:
    """
    Handles partner preference inference and priority confirmation updates
    during negotiation dialogs. Updates the original agent's properties directly.
    """

    def __init__(self, agent):
        """
        Initialize the PartnerPreferenceUpdater with reference to the agent

        Args:
            agent: Reference to the NegotiationAgent instance
        """
        self.agent = agent

    def check_priority_confirmation_and_updates(self):
        """
        Check priority confirmation and update the priority confirmation
        """
        logging.debug("> [Process] Check Priority Confirmation and Update")
        if len(self.agent.dialog_history) <= 2:
            return

        # get two latest dialog history
        recent_dialog_history = self.agent.dialog_history[-2:]
        assert recent_dialog_history[-1]['role'] == 'user'

        proc_dialog_history = self.agent.processed_dialog_history(user='ANSWER', assistant='QUESTION', num_of_latest_turns=2)

        # Import prompt_builder locally to avoid circular imports
        from prompt.prompt_build import prompt_builder

        priority_confirmation_q = prompt_builder(
            self.agent.agent_value_off_table,
            self.agent.partner_priority,
            self.agent.priority_confirmation,
            proc_dialog_history,
            prompt_type='priority_confirmation',
            verbose=self.agent.verbose
        )

        priority_confirmation_response = self.agent.call_engine(
            messages=[{"role": "user", "content": priority_confirmation_q}],
            json_parsing_check=True
        )

        priority_confirmation_response = json.loads(priority_confirmation_response["content"])
        confirmed_items = priority_confirmation_response.get("confirmed_items")

        if not confirmed_items:
            logging.info(">> No Key of 'confirmed_items'. Skipped to check the confirmed items this turn")
            return

        logging.debug(">> confirmed_items: %s", confirmed_items)
        logging.info(">> Confirmed_items (PC): %s", confirmed_items)

        # updates priorities if there is no same priority for two or more items in the response
        if not self.check_duplicate_item_priorities(priority_confirmation_response):
            for item in confirmed_items:
                confirmed_item = item["confirmed_item"]
                confirmed_priority = item["priority"]
                confirmed_meaning = item["meaning"]

                # update the priority confirmation
                self.update_priority_confirmation(confirmed_item, confirmed_priority, confirmed_meaning)

                # update item_priorities_remaining and fill in item/priorities when able
                self.agent.item_priorities_remaining = self.update_item_priorities_remaining(
                    confirmed_item, confirmed_priority, confirmed_meaning
                )

        logging.debug(" >> Item priorities remaining: %s", self.agent.item_priorities_remaining)

    def check_duplicate_item_priorities(self, priority_confirmation_response: Dict) -> bool:
        """
        Check if there are duplicate item priorities in the response

        Args:
            priority_confirmation_response: The priority confirmation response

        Returns:
            True if duplicates found, False otherwise
        """
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

    def update_item_priorities_remaining(self, confirmed_item: str, confirmed_priority: str, confirmed_meaning: str) -> Dict:
        """
        Update item priorities remaining based on confirmation

        Args:
            confirmed_item: The confirmed item
            confirmed_priority: The confirmed priority
            confirmed_meaning: The meaning of confirmation ('true', 'false', 'null')

        Returns:
            Updated item_priorities_remaining dictionary
        """
        # sets all other priorities in item_priorities_remaining for a given item to 'false'
        def set_other_priorities_false(item, confirmed_priority):
            for priority in self.agent.item_priorities_remaining[item]:
                if priority != confirmed_priority:
                    self.agent.item_priorities_remaining[item][priority] = 'false'

        # sets the confirmed priority to 'false' for all other items in item_priorities_remaining
        def set_priority_false_for_other_items(confirmed_priority):
            for item in self.agent.item_priorities_remaining:
                if item != confirmed_item:
                    self.agent.item_priorities_remaining[item][confirmed_priority] = 'false'

        # checks item_priorities_remaining if only one priority is 'true' for an item, if so it updates other items accordingly
        def update_remaining_priority(item):
            num_false = sum(1 for status in self.agent.item_priorities_remaining[item].values() if status == 'false')
            remaining_priority = next((priority for priority, status in self.agent.item_priorities_remaining[item].items() if status == 'true'), 'null')
            if num_false == 2 and remaining_priority != 'null':
                logging.debug(">> Updating item priority using remaining priorities: %s, %s", item, remaining_priority)
                self.update_item_priorities_remaining(item, remaining_priority, 'true')

        # checks if two items are false for the same priority and updates the remaining item if necessary
        def confirm_third_item_for_priority(confirmed_priority):
            false_items = [item for item, priorities in self.agent.item_priorities_remaining.items() if priorities.get(confirmed_priority) == 'false']
            if len(false_items) == 2:
                remaining_item = next(item for item in self.agent.item_priorities_remaining if item not in false_items)
                if self.agent.priority_confirmation[remaining_item] == False:
                    logging.debug(f">> Confirming third item: remaining ({remaining_item}) & {confirmed_priority}")
                    self.update_item_priorities_remaining(remaining_item, confirmed_priority, 'true')

        if confirmed_meaning == 'true':
            # updates partner_priority and priority_confirmation
            self.agent.partner_priority[confirmed_item] = confirmed_priority
            self.agent.priority_confirmation[confirmed_item] = True

            # updates item_priorities_remaining with new confirmation
            set_other_priorities_false(confirmed_item, confirmed_priority)
            set_priority_false_for_other_items(confirmed_priority)

        elif confirmed_meaning == 'false':
            # sets the priority to 'false' in item_priorities_remaining
            self.agent.item_priorities_remaining[confirmed_item][confirmed_priority] = 'false'

            # updates item_priorities_remaining if above line confirmed an item priority
            update_remaining_priority(confirmed_item)

            # checks if two items are false for the same priority, if so, updates third item
            confirm_third_item_for_priority(confirmed_priority)

        elif confirmed_meaning == 'null':
            return self.agent.item_priorities_remaining

        # final confirmation if any more updates should occur, if so, updates accordingly
        for item in self.agent.item_priorities_remaining:
            if not self.agent.priority_confirmation[item]:
                update_remaining_priority(item)

        return self.agent.item_priorities_remaining

    def update_priority_confirmation(self, confirmed_item: str, confirmed_priority: str, confirmed_meaning: str):
        """
        Update the priority confirmation based on the confirmed item and priority

        Args:
            confirmed_item: The confirmed item
            confirmed_priority: The confirmed priority
            confirmed_meaning: The meaning of confirmation
        """
        if confirmed_item == 'null' or confirmed_priority == 'null' or confirmed_meaning != 'true':
            logging.debug(">> Nothing to be updated; No priority confirmation is made")
            return

        # Check for conflicts between the priorities of items
        if any(item for item, priority in self.agent.partner_priority.items() if priority == confirmed_priority and item != confirmed_item):
            conflict_item = [item for item, priority in self.agent.partner_priority.items() if priority == confirmed_priority][0]
            logging.warning(f">> Confirmed_priority '{confirmed_priority}' is already assigned to item '{conflict_item}'; No priority confirmation is made")
            return

        # Update priority confirmation
        logging.debug("> Updating Priority Confirmation...")
        logging.debug(f"> Previous Priority Confirmation: {self.agent.priority_confirmation}")
        self.agent.priority_confirmation[confirmed_item] = True
        logging.debug(f">>> Updated Priority Confirmation: {self.agent.priority_confirmation}")

        logging.debug("> Partner's priority updated")
        logging.debug(f"> Previous partner's priority: {self.agent.partner_priority}")
        self.agent.partner_priority[confirmed_item] = confirmed_priority
        logging.debug(f">>> Updated Partner's priority: {self.agent.partner_priority}")
