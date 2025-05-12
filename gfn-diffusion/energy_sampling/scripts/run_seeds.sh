#!/bin/bash
#SBATCH --partition=long
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH -J energy_sampling
#SBATCH -o .../gfn-is/slurm_logs/%x-%j.out

cd .../gfn-is/gfn-diffusion/energy_sampling
export PYTHONPATH=".:$PYTHONPATH"

module --quiet purge
module --quiet load anaconda/3
module --quiet load cuda/12.6.0/cudnn/9.3
conda activate gfn-is
wandb login --relogin ...


ARGS=$1

for SEED in 0 1 2 3 4; do
    if [ "$SEED" = "0" ]; then
        python train.py $ARGS --seed $SEED &
    else
        python train.py $ARGS --seed $SEED --no_plot &
    fi
done
wait
