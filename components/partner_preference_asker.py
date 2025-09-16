from dataclasses import dataclass
from enum import Enum
from typing import Callable, Any, Dict, Mapping

class PromptType(str, Enum):
    PRIORITY = "generate_questions_for_priority"
    NEED_OFFER = "generate_questions_for_need_offer"

@dataclass
class PartnerPreferenceAsker:
    """
    Component for generating 'priority confirmation' questions.
    - prompt_builder: function to build the prompt string
    - call_engine: function to call the LLM/engine (expects messages=[{role, content}])
    - logger: logging module or logger object
    - verbose: passed directly to prompt_builder
    """
    prompt_builder: Callable[..., str]
    call_engine: Callable[..., Dict[str, Any]]
    logger: Any = None
    verbose: bool = False

    def ask(
        self,
        *,
        agent_value_off_table: Any,
        partner_priority: Mapping[str, Any],
        priority_confirmation: Mapping[str, bool],
        item_priorities_remaining: Mapping[str, Mapping[str, str]],
        dialog_history_processed: str,
        is_partner_priority_consistent: bool,
        ask_for_need_offer: bool = False,
    ) -> str:
        """Generate a question for confirming partner priorities."""
        if self.logger:
            self.logger.debug("> [Process] Generate Question for Priority Confirmation")

        # Choose prompt type based on whether we are asking for a need offer
        prompt_type = PromptType.NEED_OFFER if ask_for_need_offer else PromptType.PRIORITY
        inconsistency_case = "NO" if is_partner_priority_consistent else "YES"

        # Build unconfirmed item descriptions
        generate_question_q_output = self._output_unconfirmed_items(
            priority_confirmation, item_priorities_remaining
        )

        # Build the final prompt
        generate_question_q = self.prompt_builder(
            agent_value_off_table,
            partner_priority,
            generate_question_q_output,
            dialog_history_processed,
            partner_prior_inconsistency=inconsistency_case,
            prompt_type=prompt_type.value,
            verbose=self.verbose,
        )

        # Call engine to generate question text
        generated = self.call_engine(messages=[{"role": "user", "content": generate_question_q}])
        if self.logger:
            self.logger.debug(">> Generated question for priority confirmation : %s", generated)

        # Normalize whitespace
        content = generated["content"]
        return content.replace("\n", " ").replace("\t", " ")

    @staticmethod
    def _output_unconfirmed_items(
        priority_confirmation: Mapping[str, bool],
        item_priorities_remaining: Mapping[str, Mapping[str, str]],
    ) -> str:
        """
        Build text output describing unconfirmed items:
        - For each item not yet confirmed, list remaining priorities
          that are still marked as 'true'.
        """
        output_list = []
        for item, confirmed in priority_confirmation.items():
            if not confirmed:
                remaining = item_priorities_remaining.get(item, {})
                remaining_priorities = [
                    priority for priority, status in remaining.items() if status == 'true'
                ]
                remaining_priorities_str = ', '.join(remaining_priorities)
                output_list.append(
                    f"The item, {item}, could be one of the following priorities: {remaining_priorities_str}"
                )
        return '\n'.join(output_list)
