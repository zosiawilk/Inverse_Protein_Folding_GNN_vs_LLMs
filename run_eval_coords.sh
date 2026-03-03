#!/bin/bash
#SBATCH -J eval_coords
#SBATCH -A MLMI-zww20-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mail-type=NONE
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

export PATH=/rds/project/rds-xyBFuSj0hm0/MLMI2.M2025/miniconda3/envs/mlmi2/bin:$PATH

mkdir -p /home/zww20/rds/hpc-work/GDL/PiFold-main/logs
JOBID=$SLURM_JOB_ID

python -u /home/zww20/rds/hpc-work/GDL/PiFold-main/scripts/eval_llama_coords.py \
  > /home/zww20/rds/hpc-work/GDL/PiFold-main/logs/eval_coords.$JOBID 2>&1
