accelerate launch --main_process_port 29505 --num_processes 2 --num_machines 1 largemodelfinetuning_mseplusceloss.py --large_model tinyllama --kernel_size 7 --batch_size 64 --debug 
accelerate launch --main_process_port 29505 --num_processes 2 --num_machines 1 largemodelfinetuning_mseplusceloss.py --large_model tinyllama --kernel_size 7 --batch_size 64 --debug 



# accelerate launch --main_process_port 29503 --num_processes 1 --num_machines 1 largemodelfinetuning_mseplusceloss.py --use_pretrained_small_model --finetuned_small_model_checkpoint /home/beidic/yangzho6/model_checkpoints/llama-160m_deciphering_openallama3b_setting0_070600 --large_model openllama3b --kernel_size 7 
# accelerate launch --main_process_port 29502 --num_processes 2 --num_machines 1 largemodelfinetuning_mseplusceloss.py --large_model tinyllama --kernel_size 7 --resume_from_checkpoint /home/yangzho6/model_checkpoints/largemodeltinyllama_6bb5600_687874/checkpoint-1000 
# accelerate launch --main_process_port 29502 --num_processes 2 --num_machines 1 largemodelfinetuning_ce.py --use_pretrained_small_model --finetuned_small_model_checkpoint /home/yangzho6/model_checkpoints/llama-160m_deciphering_openallama3b_setting0_070600 --large_model openllama3b --kernel_size 7 
# accelerate launch --main_process_port 29503 --num_processes 1 --num_machines 1 largemodelfinetuning_ce.py --large_model openllama3b --kernel_size 7 