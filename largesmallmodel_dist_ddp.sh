# accelerate launch --main_process_port 29501 --num_processes 2 --num_machines 1 largemodelfinetuning_ce.py --use_pretrained_small_model --finetuned_small_model_checkpoint /home/yangzho6/model_checkpoints/llama-160m_deciphering_openallama3b_setting0_070600 --large_model openllama3b --kernel_size 7 
accelerate launch --main_process_port 29502 --num_processes 2 --num_machines 1 largemodelfinetuning_mseplusceloss.py --use_pretrained_small_model --finetuned_small_model_checkpoint /home/beidic/yangzho6/model_checkpoints/llama-160m_deciphering_openallama3b_setting0_070600 --large_model openllama3b --kernel_size 7 
# accelerate launch --main_process_port 29502 --num_processes 2 --num_machines 1 largemodelfinetuning_ce.py --use_pretrained_small_model --finetuned_small_model_checkpoint /home/yangzho6/model_checkpoints/llama-160m_deciphering_openallama3b_setting0_070600 --large_model openllama3b --kernel_size 7 