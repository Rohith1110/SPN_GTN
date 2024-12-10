#!/bin/bash

# Define parameters
SOLVE_PROXY_EPOCHS=1000
REFINE_EPOCHS=1000
MODEL_TYPE="GCN2"
SEED=1

# Timestamp for unique checkpoint filenames
TIMESTAMP=$(date +"%m-%d_%H-%M-%S")


# Loop through different PPI datasets
for PPI_SIZE in 10
do
    # Loop through label IDs from 0 to 120
    for LABEL_ID in {0..120}
    do
        DATASET="ppi-${PPI_SIZE}-${LABEL_ID}"
        
        echo "============================================"
        echo "Starting refinement for PPI-${PPI_SIZE}, Label ID: ${LABEL_ID}"
        echo "============================================"
        
        # --- Run Refine with Proxy ---
        echo "Running refine.py with proxy"

        python refine.py \
            --dataset ${DATASET} \
            --solve-proxy-epochs ${SOLVE_PROXY_EPOCHS} \
            --solve-proxy-node-lr 5e-3 \
            --solve-proxy-edge-lr 1e-3 \
            --solve-proxy-eval-every 100 \
            --refine-epochs ${REFINE_EPOCHS} \
            --refine-node-lr 1e-5 \
            --refine-edge-lr 1e-5 \
            --refine-eval-every 100 \
            --use_gt_layer \
            --num_transformer_layers 1 \
            --transformer_out_dim 256 \
            --transformer_num_heads 64 \
            --transformer_dropout_prob 0.05 \
            --transformer_batch_norm \
            --transformer_residual \
            ${MODEL_TYPE} \
        
        # --- Run Refine without Proxy ---
        echo "Running refine.py without proxy"

        python refine.py \
            --dataset ${DATASET} \
            --no-proxy \
            --refine-epochs ${REFINE_EPOCHS} \
            --refine-node-lr 1e-5 \
            --refine-edge-lr 1e-5 \
            --refine-eval-every 100 \
            --use_gt_layer \
            --num_transformer_layers 1 \
            --transformer_out_dim 256 \
            --transformer_num_heads 64 \
            --transformer_dropout_prob 0.05 \
            --transformer_batch_norm \
            --transformer_residual \
            --no-log-softmax ${MODEL_TYPE}
        
        echo "Submitted refinement jobs for PPI-${PPI_SIZE}, Label ID: ${LABEL_ID}"
    done
done

echo "All refinement jobs have completed."