accelerate launch --main_process_port 29512 --num_processes 2 --num_machines 1 weird_training_addinglarge.py --experiment_setting setting0 --kernel_size 7 --model_name tinyllama --use_large_model --autoregressive_first_element 