import sys
import time
import re
import json
import os
import logging
import pickle
import numpy as np

from datetime import datetime
from copy import deepcopy
from typing import List, Dict
from inspect import cleandoc as dedent

def load_json(file_path):
    """load and return JSON file as a dictionary"""
    with open(file_path, 'r') as file:
        loaded_json = json.load(file)
    return loaded_json


def cache_results(save_path, data):
    directory = os.path.dirname(save_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(save_path, 'wb') as f:
        pickle.dump(data, f)


def get_cached_results(cached_file_path):
    with open(cached_file_path, 'rb') as f:
        data = pickle.load(f)
        return data


def validate_offer(offer):
    if not offer:
        return False
    if isinstance(offer, dict):
        if all([ x in ['food', 'water', 'firewood'] for x in offer.keys() ]):
            for v in offer.values():
                if v > 3 and v < 0:
                    return False
                return True
    print(">> ! Invalid Offer format:", offer)
    return False



def wprint(s, fd=None, verbose=True):
    if(fd is not None): fd.write(s + '\n')
    if(verbose): print(s)
    return

class WPrinter:
    def __init__(self, verbose=True):
        self.verbose = verbose

    def set_verbose(self, verbose):
        self.verbose = verbose

    def wprint(self, s, fd=None):
        if fd is not None:
            fd.write(s + '\n')
        if self.verbose:
            print(s)

class Logger(object):
    def __init__(self, log_file, verbose=True):
        self.terminal = sys.stdout
        self.log = open(log_file, "w")
        self.verbose = verbose

        self.write("All outputs written to %s" % log_file)
        return

    def write(self, message):
        self.log.write(message + '\n')
        if(self.verbose): self.terminal.write(message + '\n')

    def flush(self):
        pass

def compute_time(start_time):
    return (time.time() - start_time) / 60.0


def is_resolved(utt: str) -> bool:
    """
    Given an utterance (raw output from LLM), check if the utterance ends with some form of "RESOLVE" tag

    utt: (String) raw LLM output

    Outputs a True if utterance ends with "RESOLVE"
    """
    utt = utt.strip() # remove trailing and leading whitespace
    return re.search(r"\bRESOLVE[D]*", utt) is not None

def return_non_verbal(utt: str) -> List[str]:
    """
    Given an utterance (raw output from LLM), return all the nonverbal actions, assuming prompt listed
    nonverbal actions like "abdjd (NON VERBAL ACTION)"

    utt: (String) raw LLM output

    Outputs a list of strings of nonverbal actions without the enclosing parentheses
    """
    non_verbals = []
    for elem in re.findall(r"\([A-Z\s]+\)", utt):
        non_verbals.append(elem[1:-1])
    return non_verbals

def return_text_only(utt: str) -> str:
    """
    Given an utterance (raw output from LLM), return text with no nonverbal actions and resolution tags

    utt: (String) raw LLM output

    Outputs a string without any special tags
    """

    utt = re.sub(r"\([A-Z\s]+\)", "", utt)
    utt = re.sub(r"\bRESOLVE[D]*", "", utt)
    utt = utt.strip()
    return utt

def load_txt_file(file):
    with open(file, 'r') as f:
        return f.read()

def prompt_build(personality, stance, verbose=False):

    assert personality in ['extrovert', 'introvert']
    assert stance in ['in_person', 'remote']

    personality_prompt = load_txt_file(f'prompt/personality_{personality}.txt')
    stance_prompt = load_txt_file(f'prompt/stance_{stance}.txt')
    instruction_prompt = load_txt_file(f'prompt/instruction_{personality}.txt')
    action_prompt = load_txt_file(f'prompt/action_list.txt')
    prompt=f"""
    {personality_prompt}
    ### Scenario ###
    {stance_prompt}
    ###Instruction###
    {instruction_prompt}
    ### Action List ###
    {action_prompt}
    Please Start Conversation with your partner!
    """
    prompt = dedent(prompt)

    if verbose:
        print("=" * 50)
        print("Personality: ", personality)
        print("Stance: ", stance)
        print("--------")
        print(prompt)
        print("=" * 50)
    return prompt

def save_dict_to_json(data, file_path):
    """
    Save a dictionary to a JSON file
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print(f"Data has been successfully saved to {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def convert_priority_to_number(priority: Dict) -> Dict:
    """
    Convert priority string to a number
    """
    coverted_priority = {}
    for key in priority:
        if priority[key] == "high":
            coverted_priority[key] = 5
        elif priority[key] == "medium":
            coverted_priority[key] = 4
        elif priority[key] == "low":
            coverted_priority[key] = 3
        else:
            raise ValueError(f"Invalid priority value: {priority[key]}")

    return coverted_priority

def lower_key_dict(d):
    return {k.lower(): v for k, v in d.items()}

def compute_time(start_time):
    return (time.time() - start_time) / 60.0

def convert_item_cnts_partner(item, partner_value_off_table=None, only_score=False, only_cnts=False):
    """convert item counts to partner's score and intem counts
    item: List - (int, int, int, int): (score, f, w, fw) or Dict
    value_off_table: Dict - {'food': int, 'water': int, 'firewood': int}
    """
    try:
        f, w, fw = (item['food'], item['water'], item['firewood']) if isinstance(item, dict) else (item[1], item[2], item[3])
        p_f, p_w, p_fw = 3 - f, 3 - w, 3 - fw
    except Exception as e:
        print("Error: ", e)
        print("Item: ", item)
        raise ValueError("Invalid item format")

    if only_cnts:
        return {'food': p_f, 'water': p_w, 'firewood': p_fw}

    partner_score = p_f * partner_value_off_table['food'] + p_w * partner_value_off_table['water'] + p_fw * partner_value_off_table['firewood']
    return partner_score if only_score else (partner_score, p_f, p_w, p_fw)


def calculate_prob_accept(decisions: List[str]):
    """Calculate the probability of accepting the offer
    decisions: List[str] - list of decisions made by the agent, ex) ['Accept', 'Reject', 'Accept', ...]
    """
    cnt_accept = 0
    for answer in decisions:
        assert 'Accept' in answer or 'Reject' in answer, answer
        if 'Accept' in answer:
            cnt_accept += 1
    prob_accept = cnt_accept / len(decisions)
    return prob_accept

def filter_out_offer(offer_list:List, max:int, min:int=18):
    """filter out the offer list based on the max and min value"""
    filtered_offer = [x for x in offer_list if x[0] <= max and x[0] >= min]
    return filtered_offer

def check_null_value(offer:Dict):
    """Check if the offer has null value"""
    if not offer:
        return True
    for key, value in offer.items():
        if value is None or value == 'null':
            return True
    return False

def calculate_score(offer:Dict, value_off_table:Dict):
    """Calculate the score of the offer"""
    score = 0
    for key, value in offer.items():
        if value is None or value == 'null':
            raise ValueError("The offer has null value")
        score += value * value_off_table[key]
    return score

def calculate_prob_accept(decisions: List[str]):
    """Calculate the probability of accepting the
    decisions: List[str] - list of decisions made by the agent, ex) ['Accept', 'Reject', 'Accept', ...]
    """
    cnt_accept = 0
    None_cnt = 0
    for answer in decisions:
        if answer is None:
            None_cnt += 1
            continue
        assert 'Accept' in answer or 'Reject' in answer, answer
        if 'Accept' in answer:
            cnt_accept += 1
    prob_accept = cnt_accept / len(decisions)
    #print("Cnt of NONE : ", None_cnt)
    return prob_accept

def convert_priority_str_to_int(priority_dict):
    int_mapper = {"low": 3, "middle": 4, "high": 5, 'medium': 4}
    return {k: int_mapper[v] for k, v in priority_dict.items()}

def convert_priority_int_to_str(priority_dict):
    str_mapper = {3: "low", 4: "middle", 5: "high"}
    return {k: str_mapper[v] for k, v in priority_dict.items()}

def change_role_in_dialogue(dialogue:List):
    """Change the role in the dialogue"""
    new_dialogue = deepcopy(dialogue)
    for i in new_dialogue:
        if i["role"] == "user":
            i["role"] = "asssistant"
        elif i["role"] == "assistant":
            i["role"] = "user"

    return new_dialogue

def setup_logging(level=logging.INFO):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("log", exist_ok=True)
    log_filename = f"log/agent_agent_{current_time}.log"
    logging.basicConfig(
        filename=log_filename,
        filemode='w',
        format='%(asctime)s - %(message)s',
        level=level
    )
    logging.getLogger().addHandler(logging.StreamHandler())  # This line adds console output as well


def sigmoid_scale(x, alpha=7.0, mu=0.5):
    return 1 / (1 + np.exp(-alpha * (x - mu)))


def contains_all_elements(lst, n):
    return set(lst) == set(range(n))

def strategy_full_nm_mapper(stg, inverse=False):
    ### Turn-Level Strategies ###
    mapper={"LIC": "Large Initial Concession",
            "CSC": "Continued-Smaller Concessions",
            "RC": "Reciprocal Concessions",
            "LGR": "Logrolling",
            "MGF": "Mutual Gain Focus",
            "FC": "First Concession",
            "AIO": "Aggressive Initial Offer",
            "AEO": "Aggressive Early Offers",
            "REO": "Response to Extreme Offer",
            "NCR": "No Concession Response",
            "RNC": "Reject Negative Concession ",
            }

    inverse_mapper = {v.lower(): k.upper() for k, v in mapper.items()}
    if inverse:
        return inverse_mapper.get(stg.lower(), stg)
    return mapper.get(stg.lower(), stg)


def create_capped_window(value, window_size=0.3):
    # Step 1: Create a window
    lower_bound = value - window_size
    upper_bound = value + window_size

    # Step 2: Cap the values between 0 and 1
    lower_bound = max(0, lower_bound)
    upper_bound = min(1, upper_bound)

    # Step 3: Round to two decimal places
    lower_bound = round(lower_bound, 1)
    upper_bound = round(upper_bound, 1)

    # Step 4: Split the range into intervals of 0.1 and generate a list
    split_values = [round(x, 1) for x in np.arange(lower_bound, upper_bound + 0.1, 0.1)]

    return split_values


def sync_partner_priorirty_confimation(partner_priority, partner_confirmation):
    for key, value in partner_priority.items():
        if value == 'null':
            if partner_confirmation[key] is True:
                partner_confirmation[key] = False
                print("Updated partner confirmation for ", key)


# opposite priority setting
def set_opposite_priority(priority, string_return=True):
    opposite_priority= {}
    for k, v in priority.items():
        assert k in ['food', 'water', 'firewood'], f"Invalid key: {k}"
        assert v in [3,4,5], f"Invalid value: {v}"
        opposite_priority[k] = 8 - v

    if not string_return:
        return opposite_priority
    return convert_priority_int_to_str(opposite_priority)


def set_inital_partner_priority(partner_priority, confirmaiton, agent_prirority):
    int_value_mapper = {3: "low", 4: "middle", 5: "high"}
    str_value_mapper = {"low": 3, "middle": 4, "high": 5}
    partner_prioriry = deepcopy(partner_priority)
    target_value = [3,4,5]
    target_item= []
    for key, value in partner_prioriry.items():
        if not confirmaiton[key]:
            target_item.append(key)
        else:
            target_value.remove(str_value_mapper[value])

    if len(target_item) == 3:
        partner_prioriry = set_opposite_priority(agent_prirority) # set opposite priority
    elif len(target_item) == 2: # set opposite priority for the item that is not confirmed
        target_item1, target_item2 = target_item
        if agent_prirority[target_item1] >  agent_prirority[target_item2]:
            partner_prioriry[target_item1] = int_value_mapper[min(target_value)]
            partner_prioriry[target_item2] = int_value_mapper[max(target_value)]
        elif agent_prirority[target_item1] <  agent_prirority[target_item2]:
            partner_prioriry[target_item1] = int_value_mapper[max(target_value)]
            partner_prioriry[target_item2] = int_value_mapper[min(target_value)]
        else:
            raise ValueError("Invalid priority setting")
    elif len(target_item) == 1:
        target_item1 = target_item[0]
        partner_prioriry[target_item1] = int_value_mapper[target_value[0]]

    return partner_prioriry


def filter_pareto_optimal(points):
    """
    Filters Pareto optimal points from a list of tuples (my_score, opponent_score).

    Args:
        points (list of tuples): List of scores (my_score, opponent_score).

    Returns:
        list of tuples: Pareto optimal points.
    """
    pareto_optimal = []

    for i, (x1, y1) in enumerate(points):
        dominated = False
        for j, (x2, y2) in enumerate(points):
            if i != j and x2 >= x1 and y2 >= y1 and (x2 > x1 or y2 > y1):
                # If another point dominates the current point
                dominated = True
                break
        if not dominated:
            pareto_optimal.append((x1, y1))

    return pareto_optimal


def load_json(file):
    with open(file, 'r') as f:
      out= json.load(f)
    return out

def write_json(file, data):
    with open(file, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f'{file} saved!')


def convert_to_gemini_messages(messages):
    converted_messages = []

    assert len(messages) < 3, "Gemini only supports 1 message at a time"
    system_instruction=""
    user_text=""
    for msg in messages:
        if msg["role"] == "system":
            system_instruction = msg["content"]

        elif msg["role"] == "user":
            user_text = msg["content"]

        elif msg["role"] == "assistant":
            raise ValueError("Gemini does not support assistant role in the message")

    return system_instruction, user_text
