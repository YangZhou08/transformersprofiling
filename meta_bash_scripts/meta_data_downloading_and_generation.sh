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
pip install -e . 
which python 
cd /data/home/beidic/yang/ 
git lfs version 
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4 
cd c4 
for i in {00000..00024}
do
    git lfs pull --include "en/c4-train.$i-of-01024.json.gz"
    gzip -dk "en/c4-train.$i-of-01024.json.gz" 
    mv "en/c4-train.$i-of-01024.json" "en/c4_file$i.json" 
done
cd /data/home/beidic/yang/tran
