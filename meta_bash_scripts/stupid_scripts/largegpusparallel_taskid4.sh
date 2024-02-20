#!/bin/bash
## job name
#SBATCH --job-name=yangzho6 
## filename for job standard output (stdout)
## %j is the job id, %u is the user id

#SBATCH --output=/fsx-storygen/beidic/yang/log/log-%j.out
## filename for job standard error output (stderr)
#SBATCH --error=/fsx-storygen/beidic/yang/log/log-%j.err

#SBATCH --time=4:00:00

## partition name
#SBATCH --partition=learnfair,learnlab,storygen
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
source activate base
conda activate zoo-torch20
cd /fsx-storygen/beidic/yang/transformersprofiling 
pip install termcolor 
pip install -e . 
which python 

for i in 1 2 3 4 5 6 7
do 
    CUDA_VISIBLE_DEVICES=$i python bigmodeldatasetgeneration_largegpus.py --kernel_size 7 --model_name tinyllama --path_d $((i - 1)) --batch_size 128 --task_id 4 & 
done 
# CUDA_VISIBLE_DEVICES=1 python bigmodeldatasetgeneration_largegpus.py --kernel_size 7 --model_name tinyllama --path_d 1 --batch_size 128 --task_id 0 & 

wait 
