### Run default

GMM40_T_SCALE=100.0

for LOSS in tb pis logvar db fldb subtb_chunk flsubtb_chunk; do  # tb pis logvar db fldb subtb_chunk flsubtb_chunk subtb flsubtb
    for ENERGY_NAME in gmm40 manywell; do  # gmm40 manywell
        if [ "$ENERGY_NAME" = "gmm40" ]; then
            T_SCALE=$GMM40_T_SCALE
        else
            T_SCALE=1.0
        fi

        if [ "$ENERGY_NAME" = "gmm40" ]; then
            NDIM=2
        elif [ "$ENERGY_NAME" = "manywell" ]; then
            NDIM=32
        fi

        for T in 20 50 100; do  # 20 50 100
            for BUFFER_PRIORITIZATION in normalized_iw none; do  # normalized_iw none

                # TB, logvar
                if [ "$LOSS" = "tb" ] || [ "$LOSS" = "logvar" ]; then
                    sbatch scripts/default_${LOSS}.sh $ENERGY_NAME $NDIM $BUFFER_PRIORITIZATION $T_SCALE $T

                # PIS
                elif [ "$LOSS" = "pis" ]; then
                    sbatch scripts/default_${LOSS}.sh $ENERGY_NAME $NDIM $T_SCALE $T

                # DB
                elif [ "$LOSS" = "db" ]; then
                    sbatch scripts/default_${LOSS}.sh $ENERGY_NAME $NDIM $BUFFER_PRIORITIZATION $T_SCALE $T

                # FL-DB
                elif [ "$LOSS" = "fldb" ]; then
                    LR_FLOW=0.001
                    sbatch scripts/default_${LOSS}.sh $ENERGY_NAME $NDIM $BUFFER_PRIORITIZATION $T_SCALE $T $LR_FLOW

                # SubTB
                elif [ "$LOSS" = "subtb" ]; then
                    sbatch scripts/default_${LOSS}.sh $ENERGY_NAME $NDIM $BUFFER_PRIORITIZATION $T_SCALE $T

                # FL-SubTB
                elif [ "$LOSS" = "flsubtb" ]; then
                    LR_FLOW=0.001
                    sbatch scripts/default_${LOSS}.sh $ENERGY_NAME $NDIM $BUFFER_PRIORITIZATION $T_SCALE $T $LR_FLOW

                # SubTB chunking
                elif [ "$LOSS" = "subtb_chunk" ]; then
                    N_CHUNKS=10
                    sbatch scripts/default_${LOSS}.sh $ENERGY_NAME $NDIM $BUFFER_PRIORITIZATION $T_SCALE $T $N_CHUNKS

                # FL-SubTB chunking
                elif [ "$LOSS" = "flsubtb_chunk" ]; then
                    LR_FLOW=0.001
                    N_CHUNKS=10
                    sbatch scripts/default_${LOSS}.sh $ENERGY_NAME $NDIM $BUFFER_PRIORITIZATION $T_SCALE $T $LR_FLOW $N_CHUNKS

                else
                    echo "Invalid loss: $LOSS"

                fi
            done
        done
    done
done
