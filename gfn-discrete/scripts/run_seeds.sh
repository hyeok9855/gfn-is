#!/bin/bash
#SBATCH --partition=long
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH -J gfn-discrete
#SBATCH -o .../gfn-is/slurm_logs/%x-%j.out

cd .../gfn-is/gfn-discrete
export PYTHONPATH=".:$PYTHONPATH"

module --quiet purge
module --quiet load anaconda/3
module --quiet load cuda/12.6.0/cudnn/9.3
conda activate gfn-is-discrete

# Make wandb directory if it doesn't exist
export WANDB_DIR=.../gfn-is/gfn-discrete/
mkdir -p $WANDB_DIR
wandb login --relogin ...


ARGS=$1
START_SEED=${2:-0}
END_SEED=${3:-0}

for SEED in $(seq $START_SEED $END_SEED); do
    python runexpwb.py $ARGS --seed $SEED &
done
wait
