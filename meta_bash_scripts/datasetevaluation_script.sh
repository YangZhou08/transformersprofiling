#!/bin/bash
## job name
#SBATCH --job-name=yangzho6 
## filename for job standard output (stdout)
## %j is the job id, %u is the user id

#SBATCH --output=/fsx-storygen/beidic/yang/log/log-%j.out
## filename for job standard error output (stderr)
#SBATCH --error=/fsx-storygen/beidic/yang/log/log-%j.err

#SBATCH --time=2:00:00 

## partition name
#SBATCH --partition=learnfair
## number of nodes
#SBATCH --nodes=1

## number of tasks per node
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-node=8
#SBATCH --no-requeue
## SBATCH --array=0-11 # 12 jobs in total 

source /data/home/beidic/.bashrc
source /data/home/beidic/miniconda/etc/profile.d/conda.sh 
# source activate base
conda activate zoo-torch20
cd /fsx-storygen/beidic/yang/transformersprofiling 
pip install termcolor 
pip install -e . 
which python 

export WANDB_API_KEY=fbb26fc8718b8e58d743b5cdcabaa2396656f773 
wandb login 

which python 

CUDA_VISIBLE_DEVICES=0 python Evaluation_only_script.py --model_name tinyllama --kernel_size 7 --loading_from_checkpoint YangZhoumill/llama_160m_deciphering_tinyllama_setting0_01da4cb_hf --experiment_setting setting0 --task_id 0 > /fsx-storygen/beidic/yang/log/smallmodel_evaluation_s0_0.txt 
CUDA_VISIBLE_DEVICES=1 python Evaluation_only_script.py --model_name tinyllama --kernel_size 7 --loading_from_checkpoint YangZhoumill/llama_160m_deciphering_tinyllama_setting0_01da4cb_hf --experiment_setting setting0 --task_id 1 > /fsx-storygen/beidic/yang/log/smallmodel_evaluation_s0_1.txt 
CUDA_VISIBLE_DEVICES=2 python Evaluation_only_script.py --model_name tinyllama --kernel_size 7 --loading_from_checkpoint YangZhoumill/llama_160m_deciphering_tinyllama_setting0_01da4cb_hf --experiment_setting setting0 --task_id 6 > /fsx-storygen/beidic/yang/log/smallmodel_evaluation_s0_6.txt 
CUDA_VISIBLE_DEVICES=3 python Evaluation_only_script.py --model_name tinyllama --kernel_size 7 --loading_from_checkpoint YangZhoumill/llama_160m_deciphering_tinyllama_setting0_01da4cb_hf --experiment_setting setting0 --task_id 7 > /fsx-storygen/beidic/yang/log/smallmodel_evaluation_s0_7.txt 
CUDA_VISIBLE_DEVICES=4 python Evaluation_only_script.py --model_name tinyllama --kernel_size 7 --loading_from_checkpoint YangZhoumill/llama-160m_deciphering_tinyllama_setting3_bfafdfa_hf --experiment_setting setting3 --task_id 0 > /fsx-storygen/beidic/yang/log/smallmodel_evaluation_s3_0.txt 
CUDA_VISIBLE_DEVICES=5 python Evaluation_only_script.py --model_name tinyllama --kernel_size 7 --loading_from_checkpoint YangZhoumill/llama-160m_deciphering_tinyllama_setting3_bfafdfa_hf --experiment_setting setting3 --task_id 1 > /fsx-storygen/beidic/yang/log/smallmodel_evaluation_s3_1.txt 
CUDA_VISIBLE_DEVICES=6 python Evaluation_only_script.py --model_name tinyllama --kernel_size 7 --loading_from_checkpoint YangZhoumill/llama-160m_deciphering_tinyllama_setting3_bfafdfa_hf --experiment_setting setting3 --task_id 6 > /fsx-storygen/beidic/yang/log/smallmodel_evaluation_s3_6.txt 
CUDA_VISIBLE_DEVICES=7 python Evaluation_only_script.py --model_name tinyllama --kernel_size 7 --loading_from_checkpoint YangZhoumill/llama-160m_deciphering_tinyllama_setting3_bfafdfa_hf --experiment_setting setting3 --task_id 7 > /fsx-storygen/beidic/yang/log/smallmodel_evaluation_s3_7.txt 
