"""
Agent-to-Agent Negotiation Simulation Framework

This module provides a comprehensive framework for conducting multi-agent negotiations
between AI agents with different LLM backends, negotiation strategies, and personalities.

Features:
- Support for multiple negotiation types (integrative, distributive, mixed)
- Strategic thinking and reasoning (STR) capabilities
- Priority consistency checking and partner modeling
- Comprehensive logging and evaluation metrics
- Flexible agent configuration and personality settings
"""

import argparse
import json
import logging
import os
import time
from collections import Counter
from datetime import datetime
from typing import Dict, List, Any, Optional

#from agent import NegotiationAgent, PartnerAgent, ModeratorAgent
from agent import NegotiationAgent, PartnerAgent, ModeratorAgent
from utils import (
    load_txt_file,
    convert_item_cnts_partner,
    calculate_score,
    compute_time,
    setup_logging,
    convert_priority_str_to_int,
    save_dict_to_json
)

# Global timestamp for consistent file naming
CURRENT_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class AgentMaster:
    """
    Master controller for agent-to-agent negotiation simulations.

    Manages multiple agents and orchestrates negotiation sessions with comprehensive
    evaluation metrics and logging capabilities.
    """

    def __init__(self, agents: Dict[str, Any], args: Optional[argparse.Namespace] = None):
        """
        Initialize the AgentMaster with agents and configuration.

        Args:
            agents: Dictionary of agent instances
            args: Configuration arguments from command line
        """
        self.agents = agents
        self.args = args or argparse.Namespace()

    def reset_agents(self) -> None:
        """Reset all agents to their initial state for a new simulation."""
        for agent in self.agents.values():
            agent.reset()

    def finalize_deal(
        self,
        offer_history: List[Dict[str, Any]],
        agent_value_table: Dict[str, int],
        partner_value_table: Dict[str, int],
        label: str,
        evaluation_metrics: Dict[str, Any],
        partner_inferred_priority: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Finalize the negotiation deal and calculate scores.

        Args:
            offer_history: History of offers made during negotiation
            agent_value_table: Agent's value table for items
            partner_value_table: Partner's value table for items
            label: Label describing how the deal was reached
            evaluation_metrics: Dictionary to store evaluation results
            partner_inferred_priority: Agent's inferred partner priorities
        """
        final_agreement = offer_history[-1]
        partner_agreement = convert_item_cnts_partner(final_agreement, only_cnts=True)
        agent_final_score = calculate_score(final_agreement, agent_value_table)
        partner_final_score = calculate_score(partner_agreement, partner_value_table)
        joint_score = agent_final_score + partner_final_score

        logging.info(f"[{label}] Agent's agreement: {final_agreement} ({agent_final_score})")
        logging.info(f"[{label}] Partner's agreement: {partner_agreement} ({partner_final_score})")

        # Handle walk-away scenarios
        if "WALK-AWAY" in label:
            agent_final_score, partner_final_score, joint_score = 0, 0, 0

        # Update evaluation metrics
        evaluation_metrics.update({
            "agent_score": agent_final_score,
            "partner_score": partner_final_score,
            "joint_score": joint_score,
            "win": 1 if agent_final_score > partner_final_score else 0,
            "partner_priority_prediction": (
                partner_value_table == convert_priority_str_to_int(partner_inferred_priority)
                if partner_inferred_priority else False
            ),
            "walk_away": label if "WALK-AWAY" in label else "null"
        })

    def conduct_single_simulation(
        self,
        agent_1: Any,
        agent_2: Any,
        moderator: Optional[Any] = None,
        n_round: int = 10,
        turn_level_verification: bool = False
    ) -> Dict[str, Any]:
        """
        Conduct a single negotiation simulation between two agents.

        Args:
            agent_1: Primary negotiation agent (NegotiatorAgent)
            agent_2: Partner agent
            moderator: Moderator agent for session management
            n_round: Maximum number of negotiation rounds
            turn_level_verification: Whether to pause for manual verification

        Returns:
            Dictionary containing comprehensive evaluation metrics
        """
        self.reset_agents()

        # Validation
        assert agent_1.agent_type == "NegotiatorAgent", \
            "The first agent should be NegotiatorAgent in the current setting."

        if moderator is None:
            moderator = self.agents['ModeratorAgent']

        # Log session information
        self._log_session_info(agent_1, agent_2, moderator)

        # Initialize evaluation metrics
        evaluation_metrics = {
            "agent_score": None,
            "partner_score": None,
            "joint_score": None,
            "num_rounds": None,
            "agreement": None,
            "walk_away": None,
            "win": None,
            "partner_inconsistent_behavior_count": 0
        }

        # Get value tables from global scope (preserved for compatibility)
        global agent_value_off_table, partner_agent_value_off_table

        # Main negotiation loop
        idx = 0
        total_idx = 0
        total_max_round = 40
        partner_inconsistent_behavior_count = 0

        # Initialize agent_1_response to avoid referencing before assignment
        agent_1_response = None

        while idx < n_round:
            logging.info("============ [ Round %d/%d ] ============", idx + 1, n_round)

            # Initial conversation setup
            if idx == 0:
                agent_1_response = self._handle_initial_round(agent_1, agent_2)
                idx += 1
                continue

            # Partner response
            agent_2_response = agent_2.respond(agent_1_response)
            logging.info(f"> {agent_2.agent_type}: {agent_2_response}")
            logging.info("-" * 100)

            # Check for partner acceptance
            if "ACCEPT-DEAL" in agent_2_response:
                self.finalize_deal(
                    agent_1.offer_history,
                    agent_value_off_table,
                    partner_agent_value_off_table,
                    "A2-Accept",
                    evaluation_metrics,
                    agent_1.partner_priority
                )
                self._finalize_metrics(evaluation_metrics, idx + 1, "AGREEMENT", partner_inconsistent_behavior_count)
                agent_1.dialog_history.append({"role": "user", "content": agent_2_response})
                agent_1.utterance_offer_history.append({
                    "role": "user",
                    "content": agent_2_response,
                    "offer": None
                })
                break

            # Agent response
            agent_1_response = agent_1.respond(agent_2_response)
            logging.info(f"> {agent_1.agent_type}: {agent_1_response}")
            logging.info("-" * 100)

            # Handle consistency checking
            if not agent_1.is_partner_priority_consistent:
                logging.info("**** Partner priority is NOT CONSISTENT: reset round ****")
                partner_inconsistent_behavior_count += 1
                idx = 0
                continue

            # Optional manual verification
            if turn_level_verification:
                input("Holding for turn-level response verification. Press Enter to continue...")

            # Check for agent acceptance
            if "ACCEPT-DEAL" in agent_1_response:
                self.finalize_deal(
                    agent_1.partner_offer_history,
                    agent_value_off_table,
                    partner_agent_value_off_table,
                    "A1-Accept",
                    evaluation_metrics,
                    agent_1.partner_priority
                )
                self._finalize_metrics(evaluation_metrics, idx + 1, "AGREEMENT", partner_inconsistent_behavior_count)
                break

            # Moderator status check
            check_status = moderator.check_status(agent_1.dialog_history)
            logging.info(f"( Check Status by Moderator: {check_status} )")
            logging.info("-" * 100)

            # Handle different status outcomes
            if check_status == "ACCEPT-DEAL":
                logging.info("**** [By Moderator] The negotiation is over: DEAL ****")
                self._finalize_metrics(evaluation_metrics, idx + 1, "AGREEMENT", partner_inconsistent_behavior_count)
                break
            elif check_status == "WALK-AWAY":
                if self._handle_walk_away(agent_1_response, agent_2_response, agent_1, evaluation_metrics, idx, partner_inconsistent_behavior_count):
                    break

            # Handle special round scenarios
            self._handle_special_rounds(idx, n_round, moderator, agent_1, agent_2)

            # Check for maximum rounds
            if idx == n_round - 1:
                logging.info(f"**** The negotiation is over: MAX TURNS {idx + 1}/{n_round} ****")
                self._finalize_metrics(evaluation_metrics, idx + 1, "NO-AGREEMENT", partner_inconsistent_behavior_count)
                break

            idx += 1
            total_idx += 1

            # Safety check for infinite loops
            if total_idx > total_max_round:
                logging.error(f"**** [Error] The negotiation is over: MAX TURNS {total_max_round} ****")
                raise ValueError("[Error] The negotiation exceeded maximum allowed turns")

        # Generate final logs and reports
        self._generate_final_logs(agent_1, evaluation_metrics)
        evaluation_metrics['total_round'] = total_idx

        return evaluation_metrics

    def _log_session_info(self, agent_1: Any, agent_2: Any, moderator: Any) -> None:
        """Log comprehensive session information."""
        logging.info("**** Player Information ****")
        logging.info("> Agent1: %s", agent_1.agent_type)
        logging.info("> Agent2: %s", agent_2.agent_type)
        logging.info("> Moderator: %s", moderator.agent_type)

        logging.info("**** Agent's Priority ****")
        logging.info("> Agent1 Priority: %s", agent_1.agent_value_off_table)
        logging.info("> Agent2 Priority: %s", agent_2.agent_value_off_table)
        logging.info("> Negotiation Type: %s", self.args.negotiation_type)

        logging.info("**** Player Engine ****")
        logging.info("> Agent1 Engine: %s", agent_1.engine)
        logging.info("> Agent1 STR Engine: %s", agent_1.engine_STR)
        logging.info("> Agent2 Engine: %s", agent_2.engine)
        logging.info("> Moderator Engine: %s", moderator.engine)

        logging.info("**** Partner Agent Other Prompting ****")
        logging.info("> Agent2 Other Prompting: %s", agent_2.other_prompting or "None")

        logging.info("**** Agent's w/ or wo/ STR ****")
        logging.info("> Agent1 w_STR: %s\\n\\n", agent_1.offer_proposer_w_STR)

    def _handle_initial_round(self, agent_1: Any, agent_2: Any) -> str:
        """Handle the initial conversation round."""
        logging.info("**** Starting the conversation (Agent2 start) ****")
        initial_partner_utterance = "Hello! let's start the negotiation!"
        agent_2.dialog_history.append({
            "role": "assistant",
            "content": initial_partner_utterance
        })
        agent_1_response = agent_1.respond(initial_partner_utterance)
        logging.info(f"> {agent_2.agent_type} Response: {initial_partner_utterance}")
        logging.info("-" * 100)
        logging.info(f"> {agent_1.agent_type} Response: {agent_1_response}")
        logging.info("-" * 100)
        return agent_1_response

    def _finalize_metrics(
        self,
        evaluation_metrics: Dict[str, Any],
        num_rounds: int,
        agreement_type: str,
        inconsistent_count: int
    ) -> None:
        """Finalize evaluation metrics."""
        evaluation_metrics.update({
            'partner_inconsistent_behavior_count': inconsistent_count,
            'num_rounds': num_rounds,
            'agreement': agreement_type
        })

    def _handle_walk_away(
        self,
        agent_1_response: str,
        agent_2_response: str,
        agent_1: Any,
        evaluation_metrics: Dict[str, Any],
        idx: int,
        partner_inconsistent_behavior_count: int
    ) -> bool:
        """Handle walk-away scenarios. Returns True if negotiation should end."""
        global agent_value_off_table, partner_agent_value_off_table

        if "WALK-AWAY" in agent_1_response or "WALK-AWAY" in agent_2_response:
            label = "A1-WALK-AWAY" if "WALK-AWAY" in agent_1_response else "A2-WALK-AWAY"
            logging.info("**** [By Moderator] The negotiation is over: WALK-AWAY ****")
            self.finalize_deal(
                agent_1.offer_history,
                agent_value_off_table,
                partner_agent_value_off_table,
                label,
                evaluation_metrics,
                agent_1.partner_priority
            )
            self._finalize_metrics(evaluation_metrics, idx + 1, "WALK-AWAY", partner_inconsistent_behavior_count)
            return True
        else:
            logging.info("* The moderator checked status as WALK-AWAY, but WALK-AWAY was not found in response. Continuing...")
            return False

    def _handle_special_rounds(
        self,
        idx: int,
        n_round: int,
        moderator: Any,
        agent_1: Any,
        agent_2: Any
    ) -> None:
        """Handle special round scenarios (moderation, compromise decisions)."""
        if idx == n_round - 3:
            moderate_response = moderator.moderate_conversation()
            logging.info(f"> {moderate_response}")
            agent_2.add_utterance_to_dialogue_history(moderate_response)

        if idx == n_round - 2:
            logging.info(f"**** Only one round is left before the end {idx + 1}/{n_round} ****")
            agent_1.compromised_decision_before_end = True

    def _generate_final_logs(self, agent_1: Any, evaluation_metrics: Dict[str, Any]) -> None:
        """Generate comprehensive final logs and reports."""
        has_offer = evaluation_metrics['agreement'] != "NO-AGREEMENT"

        if has_offer:
            log_whole_conversation = agent_1.process_utter_offer_history(
                agent_1.utterance_offer_history,
                agent_1.int_partner_priority,
                w_strategy=True,
                w_offer=True
            )
            log_offer_exchanges = agent_1.process_utter_offer_history(
                agent_1.utterance_offer_history,
                agent_1.int_partner_priority,
                w_strategy=True,
                w_utterance=False,
                role_user="PARTNER OFFER",
                role_assistant="YOUR OFFER"
            )
            log_concession_history = agent_1.process_utter_offer_history(
                agent_1.utterance_offer_history,
                agent_1.int_partner_priority,
                w_utterance=False,
                w_lp_params=True,
                set_turn_index=True,
                filter_None_offer=False,
                role='user',
                w_inferred_partner_priority=True,
                w_p_strategy=True
            )
        else:
            log_whole_conversation = log_offer_exchanges = log_concession_history = ""

        # Log comprehensive results
        logging.info("\\n**** Whole Conversation ****")
        logging.info(log_whole_conversation)
        logging.debug("\\n**** Whole Offer exchanges ****")
        logging.debug(log_offer_exchanges)
        logging.debug("\\n**** Concession History ****")
        logging.debug(log_concession_history)

        # Log priority prediction accuracy
        if evaluation_metrics.get('partner_priority_prediction') is False:
            global partner_agent_value_off_table
            logging.info(
                f"\\n**** GT: {partner_agent_value_off_table} | "
                f"inferred: {convert_priority_str_to_int(agent_1.partner_priority)} ****"
            )

        # Store logs in evaluation metrics
        evaluation_metrics['logs'] = {
            "whole_conversation": log_whole_conversation,
            "offer_exchanges": log_offer_exchanges,
            "concession_history": log_concession_history
        }
        evaluation_metrics['utter_level_history'] = agent_1.utterance_offer_history

    def run_experiment(
        self,
        agent_1: Any,
        agent_2: Any,
        n_exp: int = 1,
        n_round: int = 10,
        args: Optional[argparse.Namespace] = None,
        turn_level_verification: bool = False,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run multiple negotiation experiments and aggregate results.

        Args:
            agent_1: Primary negotiation agent
            agent_2: Partner agent
            n_exp: Number of experiments to run
            n_round: Number of rounds per experiment
            args: Configuration arguments
            turn_level_verification: Whether to pause for manual verification
            save_path: Path to save intermediate results

        Returns:
            Aggregated results from all experiments
        """
        start_time = time.time()

        # Initialize output structure
        output = {
            "agent_score": [],
            "partner_score": [],
            "joint_score": [],
            "num_rounds": [],
            "agreement": [],
            "walk_away": [],
            'win': [],
            'partner_priority_prediction': [],
            'logs': [],
            'utter_level_history': [],
            'partner_inconsistent_behavior_count': []
        }

        # Run experiments
        for i in range(n_exp):
            logging.info("\\n\\n\\n")
            logging.info("$$========= EXPERIMENT (%d / %d) =========$$", i + 1, n_exp)
            logging.info('Agent1 (%s) vs Agent2 (%s)\\n', agent_1.agent_type, agent_2.agent_type)

            try:
                experiment_results = self.conduct_single_simulation(
                    agent_1, agent_2,
                    n_round=n_round,
                    turn_level_verification=turn_level_verification
                )
            except Exception as e:
                logging.error(f"**** [Error] Exception: {e} ****")
                logging.error("**** Skip this session ****")
                continue

            # Aggregate results
            for key in output.keys():
                output[key].append(experiment_results.get(key))

            # Log experiment summary
            self._log_experiment_summary(experiment_results, start_time)

            # Save intermediate results
            if save_path:
                logging.info(f"==== {i}-th experiment was saved to {save_path} ====")
                save_dict_to_json(output, save_path)

            logging.info("\\n\\n")

        # Calculate aggregated statistics
        self._calculate_aggregate_stats(output)

        return output

    def _log_experiment_summary(self, results: Dict[str, Any], start_time: float) -> None:
        """Log summary of experiment results."""
        logging.info(
            "agent_score: %s | partner_score: %s | joint_score: %s | "
            "num_rounds: %s | agreement: %s | win: %s | partner_priority_prediction: %s",
            results["agent_score"],
            results["partner_score"],
            results["joint_score"],
            results["num_rounds"],
            results["agreement"],
            results["win"],
            results.get("partner_priority_prediction")
        )
        logging.info("==== End: elapsed time %.2f min ====", compute_time(start_time))

    def _calculate_aggregate_stats(self, output: Dict[str, Any]) -> None:
        """Calculate aggregate statistics from experiment results."""
        output["num_experiment"] = len(output['agent_score'])
        output_keys = list(output.keys())

        for key in output_keys:
            if key in ["num_experiment", "agreement", "logs", "utter_level_history", "walk_away"]:
                if key == "agreement":
                    output["agreement_dist"] = Counter(output[key])
                if key == "walk_away":
                    output["Rate_Walk_Away"] = (
                        len(output[key]) - output[key].count("null")
                    ) / len(output[key])
            else:
                values = [x for x in output[key] if x is not None]
                if values:
                    output[f"Avg_{key}"] = sum(values) / len(values)


def parse_experiment_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Parse experiment-related command line arguments."""
    parser.add_argument('--n_round', type=int, default=12,
                       help='Number of rounds for the negotiation')
    parser.add_argument('--n_exp', type=int, default=1,
                       help='Number of experiments')
    parser.add_argument('--agent1', type=str, default='NegotiatorAgent',
                       help='Agent1 type')
    parser.add_argument('--agent2', type=str, default='PartnerAgent',
                       help='Agent2 type')
    parser.add_argument('--moderator', type=str, default='ModeratorAgent',
                       help='Moderator type')
    parser.add_argument('--engine', type=str, default='gpt-4o-mini',
                       help='OpenAI Engine')
    parser.add_argument('--engine-OSAD', type=str, default='gpt-4o-mini',
                       help='OpenAI Engine for OSAD')
    parser.add_argument('--partner-other-prompting', type=str, default='base',
                       help='Other prompting for the partner agent')
    parser.add_argument('--engine-STR', type=str, default='gpt-4o',
                       help='OpenAI Engine for STR')
    parser.add_argument('--engine-partner', type=str, default='gpt-4o',
                       help='OpenAI Engine for partner')
    parser.add_argument('--partner-agent-personality', type=str, default='base',
                       help='Partner Agent Personality: base, greedy, fair')
    parser.add_argument('--negotiation-type', type=str, default='integrative',
                       help='Negotiation type: integrative, distributive, mixed')
    parser.add_argument('-tlrv', '--turn-level-response-verification',
                       action='store_true',
                       help='Turn-level response verification')
    parser.add_argument('--STR', action='store_true',
                       help='True or False for using STR in the Agent')
    parser.add_argument('--save', default=f'a2a_results/a2a_{CURRENT_TIME}.json',
                       help='Save the results to the file')
    return parser


def parse_agent_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Parse agent-related command line arguments."""
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose mode')
    parser.add_argument('-iv', '--inconsistency-verbose', action='store_true',
                       help='Verbose Info')
    parser.add_argument('-w1', '--weight-OSAD', type=float, default=0.5,
                       help='The weight of OSAD')
    parser.add_argument('-w2', '--weight-self-assessment', type=float, default=0.5,
                       help='The weight of self-assessment')
    parser.add_argument('--top_n', type=int, default=5,
                       help='The weight of partner priority')
    parser.add_argument('-upc', '--update-partner-priority-in-checker',
                       type=bool, default=True,
                       help='True or False for updating partner priority in checker')
    parser.add_argument('-on-cc', '--priority-consistency-checker-on',
                       type=bool, default=True,
                       help='True or False for turning on the consistency checker')
    parser.add_argument('--n-OSAD-decision', type=int, default=5,
                       help='Number of OSAD decisions for an offer')
    parser.add_argument('--fine-grained-OSAD', action='store_true',
                       help='Conduct Fine-grained OSAD')
    parser.add_argument('--n-self-assessment', type=int, default=5,
                       help='Number of OSAD decisions for an offer')
    parser.add_argument('-lp-caching', '--lp-caching-on', action='store_true',
                       help='True or False for turning on caching for LP')
    return parser


def parse_temp_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Parse temporary/experimental command line arguments."""
    parser.add_argument('--preset-partner-priority', action='store_true',
                       help='Setting the partner priority to the Agent for skipping the asking process')
    return parser


def setup_negotiation_scenario(args: argparse.Namespace) -> tuple:
    """
    Setup negotiation scenario based on negotiation type.

    Args:
        args: Parsed command line arguments

    Returns:
        Tuple of (agent_value_table, partner_value_table, partner_value_str_table)
    """
    # Base agent values
    agent_value_table = {"food": 5, "water": 4, "firewood": 3}

    # Partner values based on negotiation type
    if args.negotiation_type == "integrative":
        partner_value_table = {"food": 3, "water": 4, "firewood": 5}
    else:  # distributive or mixed
        partner_value_table = {"food": 5, "water": 4, "firewood": 3}

    # Convert to string representation
    priority_mapper = {5: "high", 4: "middle", 3: "low"}
    partner_value_str_table = {
        k: priority_mapper[v] for k, v in partner_value_table.items()
    }

    return agent_value_table, partner_value_table, partner_value_str_table


def initialize_agents(
    args: argparse.Namespace,
    agent_value_table: Dict[str, int],
    partner_value_table: Dict[str, int],
    partner_value_str_table: Dict[str, str]
) -> Dict[str, Any]:
    """
    Initialize all agents with proper configuration.

    Args:
        args: Parsed command line arguments
        agent_value_table: Agent's value table
        partner_value_table: Partner's value table
        partner_value_str_table: Partner's value table in string format

    Returns:
        Dictionary of initialized agents
    """
    # Load instruction files
    system_instruction = load_txt_file('prompt/system_instruction.txt')
    moderator_instruction = load_txt_file('prompt/moderator_instruction.txt')

    # Determine partner personality and instruction
    partner_personality = args.partner_agent_personality
    if partner_personality in ["fair", "greedy"]:
        partner_instruction = load_txt_file(f'prompt/{partner_personality}_partner_instruction.txt')
    else:
        partner_instruction = system_instruction

    partner_agent_type = f"PartnerAgent({partner_personality[0].upper()})"
    preset_partner_priority = partner_value_str_table if args.preset_partner_priority else None

    # Initialize agents
    agents = {}

    agents['NegotiatorAgent'] = NegotiationAgent(
        agent_value_off_table=agent_value_table,
        system_instruction=system_instruction,
        agent_type='NegotiatorAgent',
        engine=args.engine,
        args=args,
        preset_partner_priority=preset_partner_priority
    )

    agents['OSADAgent'] = PartnerAgent(
        agent_value_off_table=partner_value_table,
        system_instruction=system_instruction,
        agent_type='OSAD_agent',
        engine=args.engine_OSAD
    )

    # Set OSAD agent for negotiator
    agents['NegotiatorAgent'].set_OSAD_agent(agents['OSADAgent'])

    agents['ModeratorAgent'] = ModeratorAgent(
        agent_type='Moderator',
        engine=args.engine,
        system_instruction=moderator_instruction,
        trace_n_history=4,
        verbose=False
    )

    agents['PartnerAgent'] = PartnerAgent(
        agent_value_off_table=partner_value_table,
        system_instruction=partner_instruction,
        agent_type=partner_agent_type,
        engine=args.engine_partner,
        personality=partner_personality,
        other_prompting=args.partner_other_prompting
    )

    # Set OSAD agent for negotiator
    agents['NegotiatorAgent'].set_OSAD_agent(agents['OSADAgent'])

    return agents


def main():
    """Main execution function."""
    # Set up logging
    setup_logging(logging.INFO)
    #setup_logging(logging.DEBUG)

    # Parse arguments
    parser = argparse.ArgumentParser(description='Agent-Agent Simulation')
    parser = parse_experiment_args(parser)
    parser = parse_agent_args(parser)
    parser = parse_temp_args(parser)
    args = parser.parse_args()

    # Log configuration
    logging.info("**** Arguments for Simulation ****")
    logging.info(json.dumps(vars(args), indent=2))
    logging.info("\\n")

    # Set API key
    args.api_key = os.environ.get("OPENAI_API_KEY")

    # Setup negotiation scenario
    global agent_value_off_table, partner_agent_value_off_table
    agent_value_off_table, partner_agent_value_off_table, partner_value_str_table = setup_negotiation_scenario(args)

    # Initialize agents
    agents = initialize_agents(args, agent_value_off_table, partner_agent_value_off_table, partner_value_str_table)

    # Log partner priority information
    if args.preset_partner_priority:
        logging.info("\\n**** Preset Partner Priority ****")
    logging.info(f"> Partner Priority: {agents['NegotiatorAgent'].partner_priority}")
    logging.info(f"> Priority Confirmation: {agents['NegotiatorAgent'].priority_confirmation}")

    # Setup experiment
    agent1 = agents[args.agent1]
    agent2 = agents[args.agent2]
    moderator = agents[args.moderator]

    agent1.verbose = args.verbose
    agent2.verbose = args.verbose

    # Run experiments
    arena = AgentMaster(agents, args)

    # Create save directory
    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    # Execute experiments
    final_results = arena.run_experiment(
        agent_1=agent1,
        agent_2=agent2,
        n_exp=args.n_exp,
        n_round=args.n_round,
        turn_level_verification=args.turn_level_response_verification,
        save_path=args.save
    )

    # Save final results
    save_dict_to_json(final_results, args.save)
    logging.info(f"**** The final results have been saved to {args.save} ****")


if __name__ == "__main__":
    main()
