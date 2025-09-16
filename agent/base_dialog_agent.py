import json
import logging
from copy import deepcopy
from typing import Dict
from async_lib_api import completion_with_backoff, completion_with_backoff_gemini, completion_with_backoff_claude
from utils import convert_to_gemini_messages

class DialogAgent(object):
    """GPT Agent base class"""

    def __init__(self,
                 initial_dialog_history=None,  #
                 agent_type="",  # "negotiator", "partner"
                 engine="gpt-4o",
                 system_instruction=None
                 ):
        """Initialize the agent"""
        super().__init__()

        self.agent_type = agent_type
        self.engine = engine
        self.dialog_round = 0
        self.verbose = False
        self.last_prompt = ""

        self.initialize_agent(initial_dialog_history, system_instruction)
        logging.info(f"[DialogAgent] Initializing {self.agent_type} with engine ({self.engine})")

    def initialize_agent(self, initial_dialog_history, system_instruction):
        """Initialize the dialog history based on provided or default instructions."""
        if initial_dialog_history is None:
            self.validate_system_instruction(system_instruction)
            self.dialog_history = [{"role": "system", "content": system_instruction}]
            self.initial_dialog_history = [{"role": "system", "content": system_instruction}]
        else:
            self.dialog_history = deepcopy(initial_dialog_history)
            self.initial_dialog_history = deepcopy(initial_dialog_history)

    def validate_system_instruction(self, system_instruction):
        """Ensure that a system instruction is provided if no initial dialog history is given."""
        assert system_instruction is not None, "System instruction must be provided if no initial dialog history is given."

    def call_engine(self, messages, json_parsing_check=False, **kwargs):
        """Route the call to different engines"""
        engine = self.engine if kwargs.get('model') is None else kwargs.get('model')

        if kwargs.get('model'): del kwargs['model']

        if ("gpt" in self.engine):
            #logging.info("Calling GPT Engine - %s", engine)

            response = completion_with_backoff(
                model=engine,
                messages=messages,
                json_parsing_check=json_parsing_check,
                verbose=self.verbose,
                **kwargs)

            choices = response['choices']
            message = choices[0]['message'] if len(choices) == 1 else [c['message'] for c in choices]
            assert (choices[0]['message']['role'] == 'assistant')

        elif ("gemini" in self.engine):
            print("gemini-Engine")
            system_instruction, user_text = convert_to_gemini_messages(messages)
            response = completion_with_backoff_gemini(
                model=engine,
                system_instruction=system_instruction,
                user_text=user_text,
                json_parsing_check=json_parsing_check,
                verbose=self.verbose,
                **kwargs)

            choices = response['choices']
            message = choices[0]['message'] if len(choices) == 1 else [c['message'] for c in choices]
            assert (choices[0]['message']['role'] == 'assistant')
        elif ('claude' in self.engine):
            print("claude engine")
            response = completion_with_backoff_claude(
                model=engine,
                messages=messages,
                json_parsing_check=json_parsing_check,
                verbose=self.verbose,
                **kwargs)

            choices = response['choices']
            message = choices[0]['message'] if len(choices) == 1 else [c['message'] for c in choices]
            assert (choices[0]['message']['role'] == 'assistant')
        else:
            raise ValueError("Unknown engine %s" % self.engine)
        return message

    def call(self, prompt, only_w_system_instruction=False):
        """Call the agent with a prompt. Handle different backend engines in this function
        """
        if not (only_w_system_instruction):
            prompt = {"role": "user", "content": prompt}
            self.dialog_history.append(prompt)
            self.last_prompt = prompt['content']

        messages = list(self.dialog_history)
        message = self.call_engine(messages)
        self.dialog_history.append(dict(message))
        return message['content']

    def respond(self, user_input):
        """Respond to the user input"""
        #self.dialog_history.append({"role": "user", "content": user_input})
        response = self.call(user_input)
        #self.dialog_history.append({"role": "assistant", "content": response})
        return response

    def add_utterance_to_dialogue_history(self, utterance, role="user"):
        self.dialog_history.append({"role": role, "content": utterance})
        return

    def conduct_simulation(self, turns=5):
        """Conduct a simulation with the agent"""
        for i in range(turns):
            user_input = input("> Enter your response: ")
            response = self.respond(user_input)
            logging.debug("> Agent Response: %s", response)
        return

    def conduct_simulation_with_partner(self, partner_agent, turns=5):
        """Conduct a simulation with the agent and the partner agent"""
        for i in range(turns):
            user_input = input("> Enter your response: ")
            response = self.respond(user_input)
            logging.debug("> Agent Response: %s", response)
            partner_response = partner_agent.respond(response)
            logging.debug("> Partner Response: %s", partner_response)
        return

    def setup_value_off_table(self, value_off_table: Dict):
        """Setup the value offer table"""
        self.agent_value_off_table = value_off_table
        logging.debug("> Done in setting up the value offer table >> %s ", self.agent_value_off_table)
        return

    def cnt_agent_utterance(self, dialogue_history=None):
        dialog = dialogue_history or self.dialog_history
        return len([i for i in dialog if i['role'] == 'assistant'])

    @property
    def last_response(self):
        return self.dialog_history[-1]['content']

    @property
    def cnt_dialogue_history(self):
        return len([i for i in self.dialog_history if i['role'] != 'system'])


    @property
    def current_dialogue_round(self):
        return self.cnt_agent_utterance() + 1

    @property
    def history(self):
        for h in self.dialog_history:
            logging.debug('%s:  %s' % (h["role"], h["content"]))
        return

    def processed_dialog_history(self, user="PARTNER", assistant="YOU", role_change=False, dialogues=None,
                                 num_of_latest_turns=None):
        """Process the dialog history for better readability"""
        if dialogues is None:
            dialogues = [i for i in self.dialog_history if i['role'] != 'system']

        if role_change:
            user, assistant = "YOU", "PARTNER"

        if num_of_latest_turns:
            #assert num_of_latest_turns <= len(
            #    dialogues), "The number of latest dialogues should be less than the length of dialogues"
            if num_of_latest_turns <= len(dialogues):
                dialogues = dialogues[-num_of_latest_turns:]
            else:
                logging.debug("Since the number of latest dialogues is greater than the length of dialogues. We will use all dialogues.")

        proc_dialog_history = "\n".join([f"{i['role']} : {i['content']}" for i in dialogues if i['role'] != 'system'])
        return proc_dialog_history.replace("user", user).replace("assistant", assistant)

    def reset_dialog(self):
        self.dialog_history = deepcopy(self.initial_dialog_history) if self.initial_dialog_history else []
        return
