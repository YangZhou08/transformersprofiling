source /data/home/beidic/.bashrc
source /data/home/beidic/miniconda/etc/profile.d/conda.sh 
source activate base 
conda activate zoo-torch20 
cd /fsx-storygen/beidic/yang/transformersprofiling 
pip install termcolor 
pip install -e . 

export WANDB_API_KEY=fbb26fc8718b8e58d743b5cdcabaa2396656f773 
wandb login --relogin 

which python 
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 

accelerate launch --main_process_port 29505 --num_processes 8 --num_machines 1 largemodelfinetuning_mseplusceloss.py --large_model tinyllama --kernel_size 7 --batch_size 128 
