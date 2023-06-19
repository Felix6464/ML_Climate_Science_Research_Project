#!/bin/bash

#SBATCH --job-name="gkd235_first_try"
#SBATCH --time=00:5:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --partition=gpu-2080ti
#SBATCH --nodes=1
#SBATCH --output=err_lstmj.out
#SBATCH --error=err_lstm.err
#SBATCH --mail-type=END
#SBATCH --mail-user=felix.boette@student.uni-tuebingen.de

set -o errexit

pwd 

echo "Data of Job:"
scontrol show job=$SLURM_JOB_ID

echo "Run Training Script:"

python /mnt/qb/work/goswami/gkd235/ML_Climate_Science_Research_Project/LIM/neural_networks
/lstm_training.py


