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
BUFFER_PRIORITIZATION=${2:-normalized_iw}  # normalized_iw, none
SHARE_EMBEDDINGS=${3:-embshare}  # embshare, noembshare
CHUNK_RATIO=${4:-0.1}  # 0.05 0.1 0.2

if [ "$ENERGY_NAME" = "gmm40" ]; then
    N_DIM=2
    T_SCALE=100.0
elif [ "$ENERGY_NAME" = "many_well" ]; then
    N_DIM=32
    T_SCALE=1.0
fi

EXP_NAME=flsubtb-cr${CHUNK_RATIO}_buf-${BUFFER_PRIORITIZATION}_${SHARE_EMBEDDINGS}

for SEED in 0 1 2 3 4; do
    if [ "$SHARE_EMBEDDINGS" = "embshare" ]; then
        python train.py \
            --seed $SEED --energy_name $ENERGY_NAME --ndim $N_DIM --t_scale $T_SCALE --loss_type subtb --subtb_chunk_ratio $CHUNK_RATIO --eval_weighting --eval_buffer \
            --partial_energy \
            --prioritization $BUFFER_PRIORITIZATION --target_ess 0.05 --smoothing temper \
            --exp_name $EXP_NAME &
    else
        python train.py \
            --seed $SEED --energy_name $ENERGY_NAME --ndim $N_DIM --t_scale $T_SCALE --loss_type subtb --subtb_chunk_ratio $CHUNK_RATIO --eval_weighting --eval_buffer \
            --partial_energy \
            --prioritization $BUFFER_PRIORITIZATION --target_ess 0.05 --smoothing temper \
            --no_share_embeddings \
            --exp_name $EXP_NAME &
    fi
done
wait
