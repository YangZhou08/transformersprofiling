# accelerate launch --main_process_port 29501 --num_processes 2 --num_machines 1 largemodelfinetuning_mseplusceloss2.py --large_model tinyllama --kernel_size 7 --batch_size 64 --use_pretrained_small_model --finetuned_small_model_checkpoint YangZhoumill/llama_160m_deciphering_tinyllama_setting0_01da4cb_hf --ce_loss_only 
# accelerate launch --main_process_port 29506 --num_processes 2 --num_machines 1 largemodelfinetuning_mseplusceloss2.py --large_model tinyllama --kernel_size 7 --alpha 0.5 --ce_loss_only --experiment_setting setting0 --batch_size 32 --freeze_large_model --finetuned_large_model_checkpoint /home/yangzho6/model_checkpoints/largemodeltinyllama_34cb320_437372/checkpoint-4000 --lr 2e-4 
# accelerate launch --main_process_port 29506 --num_processes 2 --num_machines 1 largemodelfinetuning_mseplusceloss2.py --large_model tinyllama --kernel_size 7 --alpha 0.5 --use_mse_loss --experiment_setting setting0 --batch_size 32 --use_pretrained_small_model --finetuned_small_model_checkpoint YangZhoumill/llama_160m_deciphering_tinyllama_setting0_01da4cb_hf 
# accelerate launch --main_process_port 29509 --num_processes 2 --num_machines 1 largemodelfinetuning_mseplusceloss2.py --large_model tinyllama --kernel_size 7 --alpha 0.5 --ce_loss_only --experiment_setting setting3 --batch_size 16 --use_new_small_model_checkpoint --lr 2e-4 --debug --mixing_dataset_fla 
# accelerate launch --main_process_port 29510 --num_processes 2 --num_machines 1 weird_training_addinglarge.py --experiment_setting setting0 --kernel_size 4 --model_name tinyllama --batch_size 32 --use_large_model --autoregressive_first_element 
# accelerate launch --main_process_port 29510 --num_processes 4 --num_machines 1 weird_training_addingextralarge.py --experiment_setting setting0 --kernel_size 2 --model_name llama2_7b --use_large_model --autoregressive_first_element --batch_size 16 --usedatasettype c4 
accelerate launch --main_process_port 29510 --num_processes 1 --num_machines 1 weird_training_addinglarge.py --experiment_setting setting0 --kernel_size 7 --model_name tinyllama --use_large_model --autoregressive_first_element --batch_size 8 --debug 
# accelerate launch --main_process_port 29512 --num_processes 1 --num_machines 1 largemodelfinetuning_precisionup.py --large_model tinyllama --kernel_size 7 --alpha 0.5 --ce_loss_only --experiment_setting setting0 --use_new_small_model_checkpoint --group_compress --batch_size 64 --freeze_large_model --lr 2e-4 
# python small_model_test_ben.py --kernel_size 7 --use_pretrained_small_model --finetuned_small_model_checkpoint YangZhoumill/deciphering_7_setting0_llama-160m --large_model tinyllama --batch_size 32 --ce_loss_only --debug --use_new_small_model_checkpoint --experiment_setting setting3 

# accelerate launch --main_process_port 29509 --num_processes 2 --num_machines 1 weird_training_plain.py --model_name tinyllama --embedding_reinitialization_type xavieruniform --batch_size 10 
# accelerate launch --main_process_port 29505 --num_processes 2 --num_machines 1 largemodelfinetuning_mseplusceloss.py --large_model tinyllama --kernel_size 7 --batch_size 64 --debug 

# accelerate launch --main_process_port 29503 --num_processes 1 --num_machines 1 largemodelfinetuning_mseplusceloss.py --use_pretrained_small_model --finetuned_small_model_checkpoint /home/beidic/yangzho6/model_checkpoints/llama-160m_deciphering_openallama3b_setting0_070600 --large_model openllama3b --kernel_size 7 
# accelerate launch --main_process_port 29502 --num_processes 2 --num_machines 1 largemodelfinetuning_mseplusceloss.py --large_model tinyllama --kernel_size 7 --resume_from_checkpoint /home/yangzho6/model_checkpoints/largemodeltinyllama_6bb5600_687874/checkpoint-1000 
# accelerate launch --main_process_port 29502 --num_processes 2 --num_machines 1 largemodelfinetuning_ce.py --use_pretrained_small_model --finetuned_small_model_checkpoint /home/yangzho6/model_checkpoints/llama-160m_deciphering_openallama3b_setting0_070600 --large_model openllama3b --kernel_size 7 
# accelerate launch --main_process_port 29503 --num_processes 1 --num_machines 1 largemodelfinetuning_ce.py --large_model openllama3b --kernel_size 7 