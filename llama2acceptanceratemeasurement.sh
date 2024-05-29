#!/bin/bash

# Define the parameters
model_name="meta-llama/Llama-2-7b-chat-hf"
sparse_values=("0.25" "0.5" "0.75")
datasets=("gsm8k" "c4")
usegriffin_options=("true" "false")

# Loop through the combinations and execute the commands
for sparse in "${sparse_values[@]}"; do
    for dataset in "${datasets[@]}"; do
        for usegriffin in "${usegriffin_options[@]}"; do
            command="python example_kvcache_organization3.py --modelname $model_name --sparse $sparse --datasetname $dataset" 
            if [ "$usegriffin" = "true" ]; then
                command="$command --usegriffin"
            fi
            echo $command
            $command
        done
    done
done