accelerate launch --main_process_port 29503 --num_processes 2 --num_machines 1 weird_training_ddpstep2.py --experiment_setting setting0 --kernel_size 7 --model_name tinyllama 