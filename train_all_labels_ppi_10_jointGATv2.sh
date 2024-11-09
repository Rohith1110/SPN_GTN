#!/bin/bash

# Define parameters
EPOCHS=1000
EVAL_EVERY=100
MODEL_TYPE="GATv2"
GNN_HIDDEN_SIZES=(256 256)
GATV2_HEADS=(4 4 6)

# Loop through label IDs from 0 to 120
for LABEL_ID in {0..120}
do
    DATASET="ppi-10-${LABEL_ID}"
    OUTPUT_DIR="./results/joint_GATv2Conv_noLogSoftmax/model_ppi-10-${LABEL_ID}_seed1_${TIMESTAMP}_refine.pt"

    echo "Starting training for label ID: ${LABEL_ID}"

    python solve_proxy.py \
        --dataset ${DATASET} \
        --solve-proxy-epochs ${EPOCHS} \
        --solve-proxy-eval-every ${EVAL_EVERY} \
        --joint-model \
        ${MODEL_TYPE} \
        --GNN-hidden-sizes ${GNN_HIDDEN_SIZES[@]} \
        --GATv2-heads ${GATV2_HEADS[@]} \

    # Optional: Limit the number of parallel jobs
    # Uncomment the following lines to run 4 jobs in parallel
    # if (( $(jobs -r | wc -l) >= 4 )); then
    #     wait -n
    # fi
done

# Wait for all background jobs to finish
wait

echo "All training jobs have completed."