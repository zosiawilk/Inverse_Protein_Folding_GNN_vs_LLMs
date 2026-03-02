#!/bin/bash
#!
#SBATCH -J pifold_raw
#SBATCH -A MLMI-zww20-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --mail-type=NONE
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

source /home/${USER}/.bashrc
source /rds/project/rds-xyBFuSj0hm0/MLMI2.M2025/miniconda3/bin/activate
conda activate mlmi2

workdir="$SLURM_SUBMIT_DIR"
cd $workdir

mkdir -p logs
JOBID=$SLURM_JOB_ID

echo "JobID: $JOBID"
echo "Time: `date`"
echo "Running on: `hostname`"
echo "Directory: `pwd`"

python -u /home/zww20/rds/hpc-work/GDL/PiFold-main/main.py \
  --data_root /home/zww20/rds/hpc-work/GDL/PiFold-main/data/ \
  --data_name CATH \
  --batch_size 8 \
  --num_workers 4 \
  --epoch 100 \
  --ex_name raw_coords_run \
  --use_gpu True \
  --node_dist 0 --node_angle 0 --node_direct 0 \
  --edge_dist 1 --edge_angle 0 --edge_direct 0 \
  --patience 20 \
  > logs/out.$JOBID 2>&1
