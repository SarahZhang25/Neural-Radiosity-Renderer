#!/bin/bash
# Multi-GPU training script
# Basic usage (defaults to 2 GPUs)
# ./train_multi_gpu.sh
# Specifying the number of GPUs and a custom config
# ./train_multi_gpu.sh 4 training/train_config_46M_h5_test.yaml

# Default to 2 GPUs if not specified
NUM_GPUS=${1:-2}
# Default to the 46M config if not specified
CONFIG=${2:-"training/train_config_46M_pointnet_h5.yaml"}

# Use the conda environment that contains the project dependencies.
CONDA_ENV_NAME="neural_radiosity_renderer"
source /home/sazhang/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME"

# Construct CUDA_VISIBLE_DEVICES (e.g., 0,1 for 2 GPUs)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Force NCCL to look for NVLink / direct P2P access
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

# Enable detailed NCCL logging to inspect the topology
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,COLL


# Generate a random port between 29500 and 29999 to avoid port collisions between concurrent runs
PORT=$((29500 + RANDOM % 500))

echo "Starting multi-GPU training on $NUM_GPUS GPUs (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) using config $CONFIG on port $PORT..."
python -m torch.distributed.run --nproc_per_node=$NUM_GPUS --master_port=$PORT training/train.py --config $CONFIG
# python -m torch.distributed.run --nproc_per_node=$NUM_GPUS --master_port=$PORT training/test_run.py --config $CONFIG
