rm -rf .cache # remove cache directory for experiments for now.

# Test with the preset partner priority (not asking for the partner's preference in the beginning stage)
#python agent_agent_simulation.py --n_exp 3 --n_round 15 -w1 0.3 -w2 0.7 --preset-partner-priority -tlrv --fine-grained-OSAD
#python agent_agent_simulation.py --n_exp 1 --n_round 10 --n-OSAD-decision 3 --n-self-assessment 5 --top_n 5 -w1 0.35 -w2 0.65 --preset-partner-priority  --fine-grained-OSAD


# Test at the turn level verification
#python agent_agent_simulation.py --n_exp 3 --n_round 10 --fine-grained-OSAD -w1 0.35 -w2 0.65 --top_n 5 -tlrv

current_time=$(date "+%Y.%m.%d-%H.%M.%S")
n_exp=4
w1=0.35
w2=0.65
top_n=5
n_round=15
STR_engine='gpt-4o-mini'
partner_agent="gpt-4o-mini" # gpt-4o, gpt-4o-mini, "gemini-2.0-flash", "gemini-1.5-flash", "claude-3-5-sonnet-20241022"
partner_agent_personality="base"

#partner_other_prompting="ProCoT" # ProCoT
#save_dir=a2a_results/ProCoT
#save_dir=a2a_results/Ablation_TS_SI

save_dir=a2a_results
mkdir -p $save_dir
#negotiation_type=distributed # integrative, distributed, mixed
w_STR=True
save_file="${save_dir}/a2a_${current_time}_STR_${STR_engine}_${partner_agent_personality}_${partner_agent}_${partner_other_prompting}_${negotiation_type}_w1_${w1}_w2_${w2}_topn_${top_n}_nexp_${n_exp}_STR_${w_STR}.json"


#negotiation_types=("integrative" "distributed")
negotiation_types=("integrative")
for negotiation_type in "${negotiation_types[@]}"; do
    echo "negotiation_type: ${negotiation_type}"

    save_file="${save_dir}/a2a_${current_time}_STR_${STR_engine}_${partner_agent_personality}_${partner_agent}_${partner_other_prompting}_${negotiation_type}_w1_${w1}_w2_${w2}_topn_${top_n}_nexp_${n_exp}_STR_${w_STR}.json"

    if [ "$w_STR" = "True" ]; then
        echo "[STR | ${negotiation_type} | Nego-${STR_engine} | P-${partner_agent_personality} | P-${partner_agent}(${partner_other_prompting}) | w1: ${w1} | w2: ${w2} | top_n: ${top_n} | n_exp: ${n_exp} | save: ${save_file}]"
        python agent_agent_simulation.py --n_exp $n_exp \
        --n_round $n_round \
        --fine-grained-OSAD \
        --engine-STR $STR_engine \
        --engine-partner $partner_agent \
        -w1 $w1 \
        -w2 $w2 \
        --top_n $top_n \
        --partner-agent-personality $partner_agent_personality \
        --negotiation-type $negotiation_type \
        --save $save_file \
        --STR
    else
        echo "[No STR] [${negotiation_type} | P-${partner_agent_personality} | P-${partner_agent} | n_exp: ${n_exp} | save: ${save_file}]"
        python agent_agent_simulation.py --n_exp $n_exp --n_round $n_round --engine-partner $partner_agent --partner-agent-personality $partner_agent_personality --negotiation-type $negotiation_type --save $save_file
    fi
done
#--partner-other-prompting $partner_other_prompting \
