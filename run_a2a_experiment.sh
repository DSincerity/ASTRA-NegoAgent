#!/bin/bash
# =============================================================================
# ASTRA-NegoAgent Experiment Runner
#
# This script runs agent-to-agent negotiation experiments with ASTRA module.
# It supports various configurations including engine types, negotiation styles,
# and strategic reasoning (STR) options.
# =============================================================================

# Clean up previous cache files
rm -rf .cache

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# Timestamp for unique file naming
current_time=$(date "+%Y.%m.%d-%H.%M.%S")

# Core experiment parameters
n_exp=1                           # Number of experiments to run
n_round=15                        # Number of negotiation rounds per experiment

# Strategic reasoning weights
w1=0.35                           # Weight for OSAD (One Step Ahead Decision)
w2=0.65                           # Weight for self-assessment
top_n=5                           # Number of top offers to consider

# Engine configuration
STR_engine='gpt-4o-mini'          # LLM engine for strategic reasoning agent
partner_agent="gpt-4o-mini"       # LLM engine for partner agent
                                  # Options: "gpt-4o", "gpt-4o-mini",
                                  #          "gemini-2.0-flash", "gemini-1.5-flash",
                                  #          "claude-3-5-sonnet-20241022"

# Agent behavior configuration
partner_agent_personality="base"  # Partner personality type
                                  # Options: ["base", "greedy", "fair"]

# Advanced features
w_STR=True                        # Enable Strategic Reasoning (ASTRA module)
partner_other_prompting="base"      # Additional prompting strategy
                                  # Options: ["base", "ProCoT"]

# Negotiation types to test
negotiation_types=("integrative") # Types of negotiations to run
                                  # Options: ("integrative", "distributed", "mixed")

# Output configuration
save_dir=a2a_results              # Directory to save results

# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

# Create results directory
mkdir -p $save_dir

# Run experiments for each negotiation type
for negotiation_type in "${negotiation_types[@]}"; do
    echo "================================================================================"
    echo "Running experiment for negotiation type: ${negotiation_type}"
    echo "================================================================================"

    # Generate unique filename for this experiment
    save_file="${save_dir}/a2a_${current_time}_STR_${STR_engine}_${partner_agent_personality}_${partner_agent}_${partner_other_prompting}_${negotiation_type}_w1_${w1}_w2_${w2}_topn_${top_n}_nexp_${n_exp}_STR_${w_STR}.json"

    if [ "$w_STR" = "True" ]; then
        # Run experiment WITH Strategic Reasoning (ASTRA)
        echo "Configuration: STR ENABLED"
        echo "  - Negotiation Type: ${negotiation_type}"
        echo "  - STR Engine: ${STR_engine}"
        echo "  - Partner Engine: ${partner_agent}"
        echo "  - Partner Personality: ${partner_agent_personality}"
        echo "  - Partner Prompting: ${partner_other_prompting}"
        echo "  - Weights: w1=${w1}, w2=${w2}"
        echo "  - Top N: ${top_n}"
        echo "  - Experiments: ${n_exp}"
        echo "  - Output: ${save_file}"
        echo "--------------------------------------------------------------------------------"

        python agent_agent_simulation.py \
            --n_exp $n_exp \
            --n_round $n_round \
            --fine-grained-OSAD \
            --engine-STR $STR_engine \
            --engine-partner $partner_agent \
            -w1 $w1 \
            -w2 $w2 \
            --top_n $top_n \
            --partner-agent-personality $partner_agent_personality \
            --partner-other-prompting $partner_other_prompting \
            --negotiation-type $negotiation_type \
            --save $save_file \
            --STR
    else
        # Run experiment WITHOUT Strategic Reasoning
        echo "Configuration: STR DISABLED"
        echo "  - Negotiation Type: ${negotiation_type}"
        echo "  - Partner Engine: ${partner_agent}"
        echo "  - Partner Personality: ${partner_agent_personality}"
        echo "  - Experiments: ${n_exp}"
        echo "  - Output: ${save_file}"
        echo "--------------------------------------------------------------------------------"

        python agent_agent_simulation.py \
            --n_exp $n_exp \
            --n_round $n_round \
            --engine-partner $partner_agent \
            --partner-agent-personality $partner_agent_personality \
            --negotiation-type $negotiation_type \
            --save $save_file
    fi

    echo "Experiment for ${negotiation_type} completed."
    echo ""
done

echo "================================================================================"
echo "All experiments completed. Results saved in: ${save_dir}"
echo "================================================================================"
