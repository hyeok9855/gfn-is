### Run experiments with synthetic energies (GMM40 and ManyWell)

STARTSEED=${1:-0}
ENDSEED=${2:-4}

DEFAULT_ARGS="--energy_name manywell --ndim 32 --epochs 15000 --hidden_dim 256 --joint_layers 2 --batch_size 300 --eval_data_size 2000 --final_eval_data_size 10000"
DIFFUSION="--t_scale 1.0 --T 50 --eval_T 100 --discretizer equidistant"
LRS="--lr_fwd 0.001 --lr_Z 0.1 --use_scheduler --milestones 0.5 0.8 --gamma 0.2"
BUFFER_ARGS="--buffer_size 300000 --prefill_epochs 100 --bwd_to_fwd_ratio 2.0"
MCMC="--mcmc_type none"
COMMON_ARGS="$DEFAULT_ARGS $DIFFUSION $LRS $BUFFER_ARGS $MCMC --exp_name BSZ300EP15K"

### PIS
sbatch scripts/run_seeds.sh "$COMMON_ARGS --loss_type pis --no_use_buffer" $STARTSEED $ENDSEED

### TB (onpolicy)
sbatch scripts/run_seeds.sh "$COMMON_ARGS --loss_type tb --init_log_Z 0.0 --no_use_buffer" $STARTSEED $ENDSEED

### TB (\epsilon-expl.)
sbatch scripts/run_seeds.sh "$COMMON_ARGS --loss_type tb --init_log_Z 0.0 --no_use_buffer --epsilon 1.0" $STARTSEED $ENDSEED

### TB (IW-Training with alternating)
sbatch scripts/run_seeds.sh "$COMMON_ARGS --loss_type tb --init_log_Z 0.0 --no_use_buffer --train_weighting --alternating --target_ess 0.05" $STARTSEED $ENDSEED

### TB (normal Buffer)
sbatch scripts/run_seeds.sh "$COMMON_ARGS --loss_type tb --init_log_Z 0.0" $STARTSEED $ENDSEED

### TB (Reward-prioritized Buffer)
sbatch scripts/run_seeds.sh "$COMMON_ARGS --loss_type tb --init_log_Z 0.0 --prioritization target --buffer_sampling rank" $STARTSEED $ENDSEED

### TB (Loss-prioritized Buffer)
sbatch scripts/run_seeds.sh "$COMMON_ARGS --loss_type tb --init_log_Z 0.0 --prioritization loss --buffer_sampling rank" $STARTSEED $ENDSEED

### TB (normalized-IW-prioritized Buffer)
sbatch scripts/run_seeds.sh "$COMMON_ARGS --loss_type tb --init_log_Z 0.0 --prioritization normalized_iw --buffer_sampling systematic --target_ess 0.05" $STARTSEED $ENDSEED

### TB (IW-prioritized Buffer)
sbatch scripts/run_seeds.sh "$COMMON_ARGS --loss_type tb --init_log_Z 0.0 --prioritization iw --buffer_sampling systematic --target_ess 0.20" $STARTSEED $ENDSEED
