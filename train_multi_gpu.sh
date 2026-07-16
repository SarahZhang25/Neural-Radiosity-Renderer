#!/bin/bash
# Multi-GPU training script
# Basic usage (defaults to 2 GPUs)
# ./train_multi_gpu.sh
# Specifying the number of GPUs and a custom config
# ./train_multi_gpu.sh 4 training/train_config_46M.yaml

# Default to 2 GPUs if not specified
NUM_GPUS=${1:-2}
# Default to the 46M config if not specified
CONFIG=${2:-"training/train_config_46M.yaml"}

# Use the conda environment that contains the project dependencies.
CONDA_ENV_NAME="neural_radiosity_renderer"
source /home/sazhang/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME"

# Construct CUDA_VISIBLE_DEVICES (e.g., 0,1 for 2 GPUs)
export CUDA_VISIBLE_DEVICES=5,6

echo "Starting multi-GPU training on $NUM_GPUS GPUs (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) using config $CONFIG..."
python -m torch.distributed.run --nproc_per_node=$NUM_GPUS training/train.py --config $CONFIG
# python -m torch.distributed.run --nproc_per_node=$NUM_GPUS training/test_run.py --config $CONFIG
