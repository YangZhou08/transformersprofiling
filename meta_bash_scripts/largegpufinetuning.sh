## job name
#SBATCH --job-name=yangzho6 
## filename for job standard output (stdout)
## %j is the job id, %u is the user id

#SBATCH --output=/fsx-storygen/beidic/yang/log/log-%j.out
## filename for job standard error output (stderr)
#SBATCH --error=/fsx-storygen/beidic/yang/log/log-%j.err

#SBATCH --time=4:00:00

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
source activate base
conda activate zoo-torch20
cd /fsx-storygen/beidic/yang/transformersprofiling 
pip install -e . 

export WANDB_API_KEY=fbb26fc8718b8e58d743b5cdcabaa2396656f773 
wandb login --relogin 

which python 
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 

if [ -z "$1" ]; then 
    batch_si = 128 
else 
    batch_si = $1 
fi 

echo "batch size is $batch_si" 

# python largemodelfinetuning_mseplusceloss.py 
accelerate launch --main_process_port 29505 --num_processes 8 --num_machines 1 largemodelfinetuning_mseplusceloss.py --large_model tinyllama --kernel_size 7 --batch_size $batch_si 
