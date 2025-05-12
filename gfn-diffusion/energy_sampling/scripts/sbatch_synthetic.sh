### Run experiments with synthetic energies (GMM40 and ManyWell)

STARTSEED=0
ENDSEED=4

GMM40_T_SCALE=100.0

##### Handle energy #####
for ENERGY_NAME in gmm40 manywell; do  # gmm40 manywell
    if [ "$ENERGY_NAME" = "gmm40" ]; then
        NDIM=2
        T_SCALE=$GMM40_T_SCALE
    else
        NDIM=32
        T_SCALE=1.0
    fi

    ARGS1="--energy_name $ENERGY_NAME --ndim $NDIM --t_scale $T_SCALE"

    ##### Handle loss-specific arguments #####
    for LOSS in pis tb logvar db subtb fldb flsubtb; do  # pis tb logvar db subtb fldb flsubtb
        if [ "$LOSS" = "fldb" ]; then
            LOSS_TYPE=db
        elif [ "$LOSS" = "flsubtb" ]; then
            LOSS_TYPE=subtb
        else
            LOSS_TYPE=$LOSS
        fi

        ARGS2="--loss_type $LOSS_TYPE"

        if [ "$LOSS" = "db" ] || [ "$LOSS" = "fldb" ] || [ "$LOSS" = "subtb" ] || [ "$LOSS" = "flsubtb" ]; then
            ARGS2="$ARGS2 --plot_t_idx 25 50 75"
            if [ "$LOSS" = "fldb" ] || [ "$LOSS" = "flsubtb" ]; then
                ARGS2="$ARGS2 --partial_energy"
            fi

            if [ "$LOSS" = "subtb" ] || [ "$LOSS" = "flsubtb" ]; then
                N_CHUNKS=10
                ARGS2="$ARGS2 --subtb_n_chunks $N_CHUNKS"
            fi
        fi

        ##### Handle Miscellaneous #####
        for T in 20 50 100; do  # 20 50 100
            ARGS3="--T $T --eval_T 100"

            for LP in False True; do  # False True
                if [ "$LP" = "True" ]; then
                    ARGS4="--lp --hidden_dim 64 --flow_hidden_dim 64"
                else
                    ARGS4=""
                fi

                for TRAININGMODE in fwd both; do  # fwd both
                    ARGS5="--training_mode $TRAININGMODE"

                    # On-policy
                    if [ "$TRAININGMODE" = "fwd" ]; then
                        sbatch scripts/run_seeds.sh "$ARGS1 $ARGS2 $ARGS3 $ARGS4 $ARGS5" $STARTSEED $ENDSEED

                    # Off-policy
                    else
                        if [ "$LOSS" = "pis" ]; then
                            continue
                        fi

                        for BUFFER_PRIORITIZATION in normalized_iw target loss none; do  # normalized_iw target loss none
                            ARGS6="--prioritization $BUFFER_PRIORITIZATION --target_ess 0.05 --smoothing temper"
                            sbatch scripts/run_seeds.sh "$ARGS1 $ARGS2 $ARGS3 $ARGS4 $ARGS5 $ARGS6" $STARTSEED $ENDSEED
                        done
                    fi
                done
            done
        done
    done
done
