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
LAMBDA=${4:-2.0}
T_SCALE=${5:-1.0}

if [ "$ENERGY_NAME" = "gmm40" ]; then
    N_DIM=2
elif [ "$ENERGY_NAME" = "many_well" ]; then
    N_DIM=32
fi

EXP_NAME=subtb-l${LAMBDA}_buf-${BUFFER_PRIORITIZATION}_${SHARE_EMBEDDINGS}

for SEED in 0 1 2 3 4; do
    if [ "$SHARE_EMBEDDINGS" = "embshare" ]; then
        python train.py \
            --seed $SEED --energy_name $ENERGY_NAME --ndim $N_DIM --t_scale $T_SCALE --loss_type subtb --subtb_lambda $LAMBDA --eval_weighting --eval_buffer \
            --prioritization $BUFFER_PRIORITIZATION --target_ess 0.05 --smoothing temper \
            --exp_name $EXP_NAME &
    else
        python train.py \
            --seed $SEED --energy_name $ENERGY_NAME --ndim $N_DIM --t_scale $T_SCALE --loss_type subtb --subtb_lambda $LAMBDA --eval_weighting --eval_buffer \
            --prioritization $BUFFER_PRIORITIZATION --target_ess 0.05 --smoothing temper \
            --no_share_embeddings \
            --exp_name $EXP_NAME &
    fi
done
wait
