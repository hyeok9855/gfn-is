### Run default

GMM40_T_SCALE=100.0

for LOSS in pis tb logvar db subtb fldb flsubtb; do  # pis tb logvar db subtb fldb flsubtb

    ##### Handle loss-specific arguments #####
    if [ "$LOSS" = "fldb" ]; then
        LOSS_TYPE=db
    elif [ "$LOSS" = "flsubtb" ]; then
        LOSS_TYPE=subtb
    else
        LOSS_TYPE=$LOSS
    fi

    ARGS1="--loss_type $LOSS_TYPE"

    if [ "$LOSS" = "db" ] || [ "$LOSS" = "fldb" ] || [ "$LOSS" = "subtb" ] || [ "$LOSS" = "flsubtb" ]; then
        ARGS1="$ARGS1 --plot_t_idx 25 50 75"
        if [ "$LOSS" = "fldb" ] || [ "$LOSS" = "flsubtb" ]; then
            ARGS1="$ARGS1 --partial_energy"
        fi
    fi

    if [ "$LOSS" = "pis" ]; then
        ARGS1="$ARGS1 --training_mode fwd"
    else
        ARGS1="$ARGS1 --training_mode both"
    fi

    for ENERGY_NAME in gmm40 manywell; do  # gmm40 manywell
        ##### Handle energy #####
        if [ "$ENERGY_NAME" = "gmm40" ]; then
            NDIM=2
            T_SCALE=$GMM40_T_SCALE
        else
            NDIM=32
            T_SCALE=1.0
        fi

        ARGS2="--energy_name $ENERGY_NAME --ndim $NDIM --t_scale $T_SCALE"

        ##### Handle Miscellaneous #####
        for T in 20 50 100; do  # 20 50 100
            ARGS3="--T $T --eval_T 100"

            for LP in False True; do  # False True
                if [ "$LP" = "True" ]; then
                    ARGS4="--lp --hidden_dim 64 --flow_hidden_dim 64"
                else
                    ARGS4=""
                fi

                for BUFFER_PRIORITIZATION in normalized_iw target loss none; do  # normalized_iw target loss none
                    ARGS5="--prioritization $BUFFER_PRIORITIZATION --target_ess 0.05 --smoothing temper"

                    ##### Run with algorithm-specific arguments #####
                    ARGS="$ARGS1 $ARGS2 $ARGS3 $ARGS4 $ARGS5"

                    # PIS
                    if [ "$LOSS" = "pis" ]; then
                        if [ "$BUFFER_PRIORITIZATION" = "none" ]; then
                            sbatch scripts/run_5seed.sh "$ARGS"
                        fi

                    # TB
                    elif [ "$LOSS" = "tb" ]; then
                        sbatch scripts/run_5seed.sh "$ARGS"

                    # LogVar
                    elif [ "$LOSS" = "logvar" ]; then
                        sbatch scripts/run_5seed.sh "$ARGS"

                    # DB
                    elif [ "$LOSS" = "db" ]; then
                        sbatch scripts/run_5seed.sh "$ARGS"

                    # FL-DB
                    elif [ "$LOSS" = "fldb" ]; then
                        sbatch scripts/run_5seed.sh "$ARGS"

                    # SubTB (Chunk-based)
                    elif [ "$LOSS" = "subtb" ]; then
                        N_CHUNKS=10
                        sbatch scripts/run_5seed.sh "$ARGS --subtb_n_chunks $N_CHUNKS"

                    # FL-SubTB (Chunk-based)
                    elif [ "$LOSS" = "flsubtb" ]; then
                        N_CHUNKS=10
                        sbatch scripts/run_5seed.sh "$ARGS --subtb_n_chunks $N_CHUNKS"

                    else
                        echo "Invalid loss: $LOSS"

                    fi
                done
            done
        done
    done
done
