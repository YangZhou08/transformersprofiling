accelerate launch --main_process_port 29518 --num_processes 1 --num_machines 1 weird_training_addingextralarge.py --experiment_setting setting0 --kernel_size 4 --model_name tinyllama --use_large_model --autoregressive_first_element --data_compensation --batch_size 20 