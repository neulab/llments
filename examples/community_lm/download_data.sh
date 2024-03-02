#!/bin/bash

mkdir -p data rankings output/finetuned_gpt2_2019_dem output/finetuned_gpt2_2019_repub
wget https://raw.githubusercontent.com/hjian42/CommunityLM/main/data/anes2020/anes_pilot_2020ets_csv.csv -P data/
wget https://raw.githubusercontent.com/hjian42/CommunityLM/main/output/finetuned_gpt2_2019_dem/finetuned_gpt2_group_stance_predictions.csv -P output/finetuned_gpt2_2019_dem
wget https://raw.githubusercontent.com/hjian42/CommunityLM/main/output/finetuned_gpt2_2019_repub/finetuned_gpt2_group_stance_predictions.csv -P output/finetuned_gpt2_2019_repub
