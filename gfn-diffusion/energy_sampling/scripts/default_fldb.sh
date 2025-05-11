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


ENERGY_NAME=$1  # many_well, gmm40
NDIM=$2
BUFFER_PRIORITIZATION=${3:-normalized_iw}  # normalized_iw, none
T_SCALE=${4:-1.0}
T=${5:-100}
LR_FLOW=${6:-0.01}

for SEED in 0 1 2 3 4; do
    python train.py \
        --seed $SEED --energy_name $ENERGY_NAME --ndim $NDIM --t_scale $T_SCALE --loss_type db \
        --T $T --eval_T 100 --eval_weighting --eval_buffer --plot_t_idx 25 50 75 \
        --partial_energy --lr_flow $LR_FLOW \
        --prioritization $BUFFER_PRIORITIZATION --target_ess 0.05 --smoothing temper &
done
wait
