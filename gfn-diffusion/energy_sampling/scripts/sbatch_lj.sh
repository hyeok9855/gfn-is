### Run experiments with Lennard-Jones energies (LJ-13 and LJ-55)


for STARTSEED in 0 1 2 3 4; do  # 0 1 2 3 4
    ENDSEED=$STARTSEED

    ##### Handle energy #####
    for ENERGY_NAME in lj13 lj55; do  # lj13 lj55
        if [ "$ENERGY_NAME" = "lj13" ]; then
            BATCHSIZE=64
            PREFILL=100
            ARGS1="--batch_size $BATCHSIZE --prefill $PREFILL"
        elif [ "$ENERGY_NAME" = "lj55" ]; then
            BATCHSIZE=64
            PREFILL=100
            ARGS1="--batch_size $BATCHSIZE --prefill $PREFILL --use_checkpoint --logr_lb=-1e7"
        fi

        for TSCALE in 0.1 0.2 0.5 1.0; do  # 0.1 0.2 0.5 1.0
            ARGS2="--energy_name $ENERGY_NAME --module egnn --t_scale $TSCALE --batch_size $BATCHSIZE --prefill $PREFILL"

            ##### Handle loss-specific arguments #####
            for LOSS in pis tb logvar; do  # pis tb logvar // TODO: db subtb fldb flsubtb
                if [ "$LOSS" = "fldb" ]; then
                    LOSS_TYPE=db
                elif [ "$LOSS" = "flsubtb" ]; then
                    LOSS_TYPE=subtb
                else
                    LOSS_TYPE=$LOSS
                fi

                ARGS3="--loss_type $LOSS_TYPE"

                if [ "$LOSS" = "db" ] || [ "$LOSS" = "fldb" ] || [ "$LOSS" = "subtb" ] || [ "$LOSS" = "flsubtb" ]; then
                    ARGS3="$ARGS3 --plot_t_idx 25 50 75"
                    if [ "$LOSS" = "fldb" ] || [ "$LOSS" = "flsubtb" ]; then
                        ARGS3="$ARGS3 --partial_energy"
                    fi

                    if [ "$LOSS" = "subtb" ] || [ "$LOSS" = "flsubtb" ]; then
                        N_CHUNKS=10
                        ARGS3="$ARGS3 --subtb_n_chunks $N_CHUNKS"
                    fi
                fi

                ##### Handle Miscellaneous #####
                # Note: For LJs, LP is not used since it's much worse than the default
                for T in 20 50 100; do  # 20 50 100
                    ARGS4="--T $T --eval_T 100"

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
done
