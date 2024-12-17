# Structured Proxy Network (SPN)
[Neural Structured Prediction for Inductive Node Classification](https://openreview.net/forum?id=YWNAX0caEjI).

## Introduction
SPN focuses on inductive node classification, which is a fundamental problem in graph machine learning and structured prediction. The idea of SPN is to combine **Graph Neural Networks (GNNs)** and **Conditional Random Fields (CRFs)** by parameterizing the potential functions of CRFs with GNNs.

Model learning of CRF is typically achieved through maximizing likelihood. However, this method is often unstable, epecially when the potential functions are parameterized by high-capacity neural models (e.g., GNN). In order to overcome the challenge, SPN instead optimizes a **proxy optimization problem**, which acts as a surrogate for the original maximizing likelihood problem. Solving this proxy problem yields a near-optimal solution on training graphs. Afterwards, SPN further performs **model refinement** by applying belief propagation and gradient descent for optimizing the maximizing likelihood problem.

Once a SPN is learned, it can be applied to test graphs for estimating the joint distribution of node labels. Then loopy belief propagation is used to infer node labels based on the joint distribution.

## Directory Structure
- `belief_propagation.py`: Core components of the Structured Proxy Network. Implements the CRF theta-functions and sum-/max- product belief propagation in pytorch.
- `config.py`: Command line argument parser.
- `dataset.py`: Dataset loading utilities.
- `eval_ppi_pred.py`: Evaluation PPI predictions generated by `refine.py` or `solve_proxy.py`. (The PPI dataset has 121 targets. We train separate models on them and evaluate the overall results here.)
- `refine.py`: Main entrance that, by default, trains the Structured Proxy Network with first the proxy problem (Eq. 10 of the paper) and then the CRF maximin game objective (Eq. 3 of the paper). If `--no-proxy` is specified, then will not train with the proxy problem, and only train with the maximin game objective.
- `models.py`: Implements various node and edge GNN models, and two ways to combine them with different levels of weight tying (`SeparateModel` and `JointModel`).
- `solve_proxy.py`: Main entrace that trains the Structured Proxy Network with the proxy problem (Eq. 10 of the paper).
- `train_loops.py`: Train and evaluate utilities.
- `utils.py`: Miscellaneous utilities.

## Usage
```
python {solve_proxy,refine}.py [global_params] [GNN_model] [GNN_params]
[global_params]: dataset, seed, epochs, eval_every, lr, checkpoint & log path, bp params, etc. See `config.py` for details.
    dataset: one of {ppi-[n_graphs]-[lid], cora, citeseer, pubmed, dblp}
        n_graphs: one of {1, 2, 10}
        lid: one of {0, 1, 2, ..., 120}
[GNN_model]: one of {GCN, GAT, GCN2, SAGE, CRF, DeeperGCN, GraphUNet}
[GNN_params]: GAT heads, GCN_normalize, etc. See `config.py` for details.
```

Below we provide some examples and explanations.

**1. Train an SPN-SAGE model**

When we run `refine.py`, the default behavior is doing both *proxy problem solving* (Eq. 10 of the paper) and *refinement* (with the CRF maximin game, Eq. 3 of the paper). In the proxy-solving stage, the model is trained with the proxy problem of node/edge label classification (Eq. 10). In the refinement stage, the model is trained by optimizing the maximin game of the CRF (Eq. 3).

The SPN is a CRF (Eq. 2) with potential functions parameterized by Eq. 9, where the pseudomarginals (\(\tau_s\) and \(\tau_{st}\)) are parameterized by node and edge GNNs, respectively (Eq. 7 and 8).
```bash
python refine.py --dataset cora \
    --solve-proxy-epochs 200 \
    --solve-proxy-eval-every 20 \
    --solve-proxy-node-lr 1e-2 \
    --solve-proxy-edge-lr 2e-3 \
    --refine-node-lr 1e-3 \
    --refine-edge-lr 2e-4 \
    SAGE
```

**2. Refine an SPN-GCN model without the proxy problem**

With `--no-proxy` specified, will only perform refinement, so training-related arguments should be `--refine-[xx]`.

The argument `--joint-model` means we use the same GNN encoder for the node and edge potentials (only the output layer is different).
```bash
python refine.py --dataset ppi-2-99 \
    --no-proxy \
    --refine-epochs 500 \
    --joint-model
    GCN
```

**3.  Train a CRF-GAT model**

Here we train a CRF-GAT model with the potential functions (theta-functions) parameterized directly with GNNs (rather than with Eq. 9 in the paper). We optimize the model with the maximin game of the CRF (Eq. 3). The training process is equivalent to refine a no-log-softmax SPN-GAT without the proxy problem.
```bash
python refine.py --dataset pubmed \
    --no-proxy \
    --refine-epochs 5000 \
    --refine-eval-every 50 \
    --no-log-softmax \
    GAT
```

**4. Train a GAT model**

Note that training a GNN model on the dataset is the same as training an SPN-GNN model and take the node GNN, so we can run `solve_proxy.py` to do that. The GAT architecture can be specified with the `[GNN_params]`, which follow the argument `GAT`.
```bash
python solve_proxy.py --dataset ppi-10-0 \
    --solve-proxy-epochs 400 \
    --dropout-prob 0.5 \
    GAT \
    --GNN-hidden-sizes 64 64 \
    --GAT-heads 4 4 6
```

**5. Train a GATv2 model**
```bash
python solve_proxy.py --dataset ppi-10-0 \
    --solve-proxy-epochs 400 \
    --dropout-prob 0.5 \
    GATv2 \
    --GNN-hidden-sizes 64 64 \
    --GAT-heads 4 4 6
```

**6. Train a SPN-GT joint model**
```bash
    python refine.py \
        --dataset ppi-10-0 \
        --solve-proxy-epochs 1000 \
        --solve-proxy-eval-every 100 \
        --refine-epochs 1000 \
        --refine-eval-every 100 \
        --use_gt_layer \
        --num_transformer_layers 1 \
        --transformer_out_dim 256 \
        --transformer_num_heads 64 \
        --transformer_dropout_prob 0.05 \
        --transformer_batch_norm \
        --transformer_residual \
        --joint-model \
        GCN2
```

**7. Refine a SPN-GT joint model**
```bash
    python refine.py \
        --dataset ppi-10-0 \
        --no-proxy \
        --refine-epochs 1000 \
        --refine-eval-every 100 \
        --use_gt_layer \
        --num_transformer_layers 1 \
        --transformer_out_dim 256 \
        --transformer_num_heads 64 \
        --transformer_dropout_prob 0.05 \
        --transformer_batch_norm \
        --transformer_residual \
        --joint-model \
        --no-log-softmax GCN2
```

**8. Train a CRF**

Specifying CRF as the model will automatically set the `--no-proxy` argument, so no need to specify that. In this case, we also only perform refinement, so training-related arguments should be `--refine-[xx]`.
```bash
python refine.py --dataset pubmed \
    --refine-epochs 5000 \
    --refine-eval-every 50 \
    CRF
```


This entire implementation is for SPN. We have created multiple branches with new features for SPN.
## Datasets

We provide the processed Citation datasets (Cora\*, CiteSeer\*, Pubmed\*) in the file `Citation.7z`. The dataset consists of ego-graphs extracted from the citation networks. Decompress it in the directory `./data` before running.

