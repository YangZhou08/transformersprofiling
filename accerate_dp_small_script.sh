# accelerate launch --main_process_port 29515 --num_processes 2 --num_machines 1 weird_training_ddpstep2.py --experiment_setting setting3 --kernel_size 7 --model_name tinyllama --autoregressive_first_element 
# accelerate launch --main_process_port 29515 --num_processes 2 --num_machines 1 weird_training_ddpstep2.py --experiment_setting setting0 --kernel_size 7 --model_name tinyllama 
accelerate launch --main_process_port 29515 --num_processes 1 --num_machines 1 weird_training_ddpstep.py --model_name llama2_7b --use_plain_model --kernel_size 7 --requeuepassword ckNtYeMSjNaam9MmGMMbi 
# accelerate launch --main_process_port 29512 --num_processes 2 --num_machines 1 weird_training_ddpstep.py --use_plain_model --kernel_size 7 
# accelerate launch --main_process_port 29502 --num_processes 2 --num_machines 1 weird_training_ddpstep.py --experiment_setting setting0 --kernel_size 7 --model_name tinyllama --use_plain_model 