#!/bin/bash
set -x
#SBATCH -J pifold_llama
#SBATCH -A MLMI-zww20-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --mail-type=NONE
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

source /home/${USER}/.bashrc
source /rds/project/rds-xyBFuSj0hm0/MLMI2.M2025/miniconda3/bin/activate
conda activate mlmi2

mkdir -p /home/zww20/rds/hpc-work/GDL/PiFold-main/logs
JOBID=$SLURM_JOB_ID

python -u /home/zww20/rds/hpc-work/GDL/PiFold-main/scripts/train_llama.py \
  > /home/zww20/rds/hpc-work/GDL/PiFold-main/logs/out.$JOBID 2>&1

