### Run default

# TB = 2 x 2 = 4
# DB = 2 x 2 x 2 = 8
# SubTB = 2 x 2 x 2 x 2 = 16
# Fldb = 2 x 2 x 2 = 8
# FlSubTB = 2 x 2 x 2 x 2 = 16
# SubTB chunking = 2 x 2 x 2 x 3 = 24
# FlSubTB chunking = 2 x 2 x 2 x 3 = 24
# Total # of experiments = 4 + 8 + 16 + 8 + 16 + 24 + 24 = 100

GMM40_T_SCALE=400.0

for LOSS in tb db subtb fldb flsubtb subtb_chunking flsubtb_chunking; do  # tb db subtb fldb flsubtb subtb_chunking flsubtb_chunking
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
                for SHARE_EMBEDDINGS in embshare noembshare; do  # embshare noembshare
                    sbatch scripts/default_${LOSS}.sh $ENERGY_NAME $BUFFER_PRIORITIZATION $SHARE_EMBEDDINGS $T_SCALE
                done

            # SubTB FL-SubTB
            elif [ "$LOSS" = "subtb" ] || [ "$LOSS" = "flsubtb" ]; then
                for SHARE_EMBEDDINGS in embshare noembshare; do  # embshare noembshare
                    for LAMBDA in 1.5 2.0; do
                        sbatch scripts/default_${LOSS}.sh $ENERGY_NAME $BUFFER_PRIORITIZATION $SHARE_EMBEDDINGS $LAMBDA $T_SCALE
                    done
                done

            # SubTB chunking FL-SubTB chunking
            elif [ "$LOSS" = "subtb_chunking" ] || [ "$LOSS" = "flsubtb_chunking" ]; then
                for SHARE_EMBEDDINGS in embshare noembshare; do  # embshare noembshare
                    for CHUNK_RATIO in 0.05 0.1 0.2; do  # 0.05 0.1 0.2
                        sbatch scripts/default_${LOSS}.sh $ENERGY_NAME $BUFFER_PRIORITIZATION $SHARE_EMBEDDINGS $CHUNK_RATIO $T_SCALE
                    done
                done

            else
                echo "Invalid loss: $LOSS"

            fi
        done
    done
done
