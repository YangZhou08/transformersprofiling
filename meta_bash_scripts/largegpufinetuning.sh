## job name
#SBATCH --job-name=yangzho6 
## filename for job standard output (stdout)
## %j is the job id, %u is the user id

#SBATCH --output=/data/home/beidic/yang/log-%j.out
## filename for job standard error output (stderr)
#SBATCH --error=/data/home/beidic/yang/log-%j.err

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
#SBATCH --array=0-11 # 12 jobs in total 

source /data/home/beidic/.bashrc
source /data/home/beidic/miniconda/etc/profile.d/conda.sh
source activate base
conda activate zoo-torch20
cd /data/home/beidic/yang/transformersprofiling 
pip install -e . 
which python 
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 

