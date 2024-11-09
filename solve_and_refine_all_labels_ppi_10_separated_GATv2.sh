#!/bin/bash

# Script: refine_all_labels.sh
# Description: Automates running refine.py for all PPI label IDs (0-120) with and without proxy.

# Define parameters
SOLVE_PROXY_EPOCHS=1000
REFINE_EPOCHS=1000
MODEL_TYPE="GATv2"
GNN_HIDDEN_SIZES=(256 256)
GATV2_HEADS=(4 4 6)
SEED=1

# Timestamp for unique checkpoint filenames
TIMESTAMP=$(date +"%m-%d_%H-%M-%S")

# Maximum number of parallel jobs (adjust based on your system's capabilities)
# MAX_PARALLEL=4

# Starting label ID (continue from where it stopped)
START_LABEL_ID=0

# Function to wait for background jobs if the limit is reached
# function wait_for_jobs {
#     while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
#         sleep 1
#     done
# }

# Loop through label IDs from 0 to 120

for LABEL_ID in $(seq ${START_LABEL_ID} 120)
# for LABEL_ID in {0..120}
do
    DATASET="ppi-10-${LABEL_ID}"
    
    # Define checkpoint paths
    # PROXY_CHECKPOINT="./results/joint_GATv2Conv/model_ppi-10-${LABEL_ID}_seed${SEED}_${TIMESTAMP}_proxy.pt"
    # REFINE_CHECKPOINT_WITH_PROXY="./results/joint_GATv2Conv/model_ppi-10-${LABEL_ID}_seed${SEED}_${TIMESTAMP}_refine_with_proxy.pt"
    # REFINE_CHECKPOINT_NO_PROXY="./results/joint_GATv2Conv/model_ppi-10-${LABEL_ID}_seed${SEED}_${TIMESTAMP}_refine_no_proxy.pt"
    
    echo "============================================"
    echo "Starting refinement for Label ID: ${LABEL_ID}"
    echo "============================================"
    
    # --- Run Refine with Proxy ---
    echo "Running refine.py with proxy for Label ID: ${LABEL_ID}"
    
    # wait_for_jobs  # Ensure we don't exceed MAX_PARALLEL jobs
    
    python refine.py \
        --dataset ${DATASET} \
        --solve-proxy-epochs ${SOLVE_PROXY_EPOCHS} \
        --solve-proxy-eval-every 100 \
        --refine-epochs ${REFINE_EPOCHS} \
        --refine-eval-every 100 \
        ${MODEL_TYPE} \
        --GNN-hidden-sizes ${GNN_HIDDEN_SIZES[@]} \
        --GATv2-heads ${GATV2_HEADS[@]} \
    
    # --- Run Refine without Proxy ---
    echo "Running refine.py without proxy for Label ID: ${LABEL_ID}"
    
    # wait_for_jobs  # Ensure we don't exceed MAX_PARALLEL jobs
    
    python refine.py \
        --dataset ${DATASET} \
        --no-proxy \
        --refine-epochs ${REFINE_EPOCHS} \
        --refine-eval-every 100 \
        --no-log-softmax ${MODEL_TYPE} \
        --GNN-hidden-sizes ${GNN_HIDDEN_SIZES[@]} \
        --GATv2-heads ${GATV2_HEADS[@]} \
    
    echo "Submitted refinement jobs for Label ID: ${LABEL_ID}"
done

# Wait for all background jobs to finish
wait

echo "All refinement jobs have completed."