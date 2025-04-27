### Run default

GMM40_T_SCALE=100.0

for LOSS in tb db subtb fldb flsubtb subtb_chunk flsubtb_chunk; do  # tb db subtb fldb flsubtb subtb_chunk flsubtb_chunk
    for ENERGY_NAME in gmm40 many_well; do  # gmm40 many_well
        if [ "$ENERGY_NAME" = "gmm40" ]; then
            T_SCALE=$GMM40_T_SCALE
        else
            T_SCALE=1.0
        fi

        for BUFFER_PRIORITIZATION in normalized_iw none; do  # normalized_iw none

            # TB
            if [ "$LOSS" = "tb" ]; then
                sbatch scripts/default_${LOSS}.sh $ENERGY_NAME $BUFFER_PRIORITIZATION $T_SCALE

            # DB FL-DB
            elif [ "$LOSS" = "db" ] || [ "$LOSS" = "fldb" ]; then
                sbatch scripts/default_${LOSS}.sh $ENERGY_NAME $BUFFER_PRIORITIZATION $T_SCALE

            # SubTB FL-SubTB
            elif [ "$LOSS" = "subtb" ] || [ "$LOSS" = "flsubtb" ]; then
                for LAMBDA in 1.5 2.0; do
                    sbatch scripts/default_${LOSS}.sh $ENERGY_NAME $BUFFER_PRIORITIZATION $T_SCALE $LAMBDA
                done

            # SubTB chunking FL-SubTB chunking
            elif [ "$LOSS" = "subtb_chunk" ] || [ "$LOSS" = "flsubtb_chunk" ]; then
                for CHUNK_SIZE in 10 20; do  # 10 20
                    sbatch scripts/default_${LOSS}.sh $ENERGY_NAME $BUFFER_PRIORITIZATION $T_SCALE $CHUNK_SIZE
                done

            else
                echo "Invalid loss: $LOSS"
            fi
        done
    done
done
