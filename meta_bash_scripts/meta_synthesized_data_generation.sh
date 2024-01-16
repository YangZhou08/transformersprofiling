#!/bin/bash
## SLURM scripts have a specific format. 

### Section1: SBATCH directives to specify job configuration

## job name
#SBATCH --job-name=yangzho6 
## filename for job standard output (stdout)
## %j is the job id, %u is the user id

#SBATCH --output=/data/home/beidic/yang/log-%j.out
## filename for job standard error output (stderr)
#SBATCH --error=/data/home/beidic/yang/log-%j.err

#SBATCH --time=6:00:00

## partition name
#SBATCH --partition=learnfair
## number of nodes
#SBATCH --nodes=1

## number of tasks per node
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-node=8
#SBATCH --no-requeue

source /data/home/beidic/.bashrc
source /data/home/beidic/miniconda/etc/profile.d/conda.sh
source activate base
conda activate zoo-torch20 
cd /data/home/beidic/yang/transformersprofiling 
pip install -e . 
which python 
cd /data/home/beidic/yang/transformersprofiling 

CUDA_VISIBLE_DEVICES=0 python bigmodeldatasetgeneration.py --kernel_size 7 --model_name tinyllama --path_d 0 & 
CUDA_VISIBLE_DEVICES=1 python bigmodeldatasetgeneration.py --kernel_size 7 --model_name tinyllama --path_d 1 & 
CUDA_VISIBLE_DEVICES=2 python bigmodeldatasetgeneration.py --kernel_size 7 --model_name tinyllama --path_d 2 & 

wait 