# ### Test MLE
# for SEED in 0 1 2 3 4; do
#     sbatch scripts/run_seeds.sh "--energy_name aldp --loss_type mle --epochs 30000 --hidden_dim 1024 --joint_layers 2  --eval_data_size 2000 --final_eval_data_size 100000 --no_full_eval --t_scale 1.0 --T 100 --eval_T 500 --discretizer equidistant --epsilon 0.0 --lr_fwd 0.0005 --lr_Z 0.05 --use_scheduler --milestones 0.5 0.8 --gamma 0.2 --invtemp 1.0 --mcmc_type none --exp_name DefenseFinal" $SEED $SEED
# done

LRFWD=0.0005
LRZ=0.05
EPSILON=0.0
INVTEMP=1.0

MCMCNSTEPS=500
MCMCBATCHSIZE=100
MCMCGAMMA=2.5
MCMCFREQ=500
MCMCTHINNING=1

BWDTOFWD=2.0

DEFAULT_ARGS="--energy_name aldp --loss_type tb --init_log_Z iw_elbo --epochs 30000 --hidden_dim 1024 --joint_layers 2 --eval_data_size 2000 --final_eval_data_size 100000 --no_full_eval"
DIFFUSION="--t_scale 1.0 --T 100 --eval_T 500 --discretizer equidistant --epsilon $EPSILON"
LRS="--lr_fwd $LRFWD --lr_Z $LRZ --use_scheduler --milestones 0.5 0.8 --gamma 0.2 --invtemp $INVTEMP"
MCMC="--mcmc_type md --mcmc_freq $MCMCFREQ --mcmc_n_steps $MCMCNSTEPS --mcmc_batch_size $MCMCBATCHSIZE --mcmc_burn_in 0 --mcmc_step_size 0.001 --mcmc_gamma $MCMCGAMMA --mcmc_thinning $MCMCTHINNING"
BUFFER_ARGS="--buffer_size 2000000 --bwd_to_fwd_ratio $BWDTOFWD --prefill_epochs 100"

DEFAULT_ARGS_PIS="--energy_name aldp --loss_type pis --epochs 30000 --hidden_dim 1024 --joint_layers 2 --eval_data_size 2000 --final_eval_data_size 100000 --no_full_eval"

### RUN

STARTSEED=${1:-0}
ENDSEED=${2:-4}

for SEED in $(seq $STARTSEED $ENDSEED); do
    sbatch scripts/run_seeds.sh "$DEFAULT_ARGS_PIS $DIFFUSION $LRS --no_use_buffer --exp_name DefenseFinal" $SEED $SEED
    sbatch scripts/run_seeds.sh "$DEFAULT_ARGS $DIFFUSION $LRS --no_use_buffer --exp_name DefenseFinal" $SEED $SEED
    for PRIORITIZATION in none target loss normalized_iw iw; do  # none target loss normalized_iw iw
        if [ "$PRIORITIZATION" == "iw" ]; then
            TARGETESS=0.20
        else
            TARGETESS=0.05
        fi

        if [ "$PRIORITIZATION" == "target" ] || [ "$PRIORITIZATION" == "loss" ]; then
            BUFFER_SAMPLING="rank"
        else
            BUFFER_SAMPLING="systematic"
        fi
        sbatch scripts/run_seeds.sh "$DEFAULT_ARGS $DIFFUSION $LRS $BUFFER_ARGS --prioritization $PRIORITIZATION --target_ess $TARGETESS --mcmc_type none --buffer_sampling $BUFFER_SAMPLING --exp_name DefenseFinal" $SEED $SEED
        sbatch scripts/run_seeds.sh "$DEFAULT_ARGS $DIFFUSION $LRS $MCMC $BUFFER_ARGS --prioritization $PRIORITIZATION --target_ess $TARGETESS --buffer_sampling $BUFFER_SAMPLING --exp_name DefenseFinal" $SEED $SEED
    done
done
