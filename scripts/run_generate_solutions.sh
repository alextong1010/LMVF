#!/bin/bash

# Change to repo root
cd "$(dirname "$0")/.."

# --- Configuration ---
CONFIG_FILE="configs/gen_config.yaml"
CONFIG_FILE=$(realpath "$CONFIG_FILE")
# --- End Configuration ---

# Ensure yq is available
if ! command -v yq &> /dev/null
then
    echo "Error: yq not found. Install with: conda install -c conda-forge yq"
    exit 1
fi

# Read config values using yq
GPUS_PER_NODE=$(yq -r '.gpus_per_node' "$CONFIG_FILE")
NUM_NODES=$(yq -r '.num_nodes' "$CONFIG_FILE")
MODEL_IDENTIFIER=$(yq -r '.model' "$CONFIG_FILE")

# Validate required config values
if [ -z "$MODEL_IDENTIFIER" ]; then echo "Error: Could not read 'model' from $CONFIG_FILE"; exit 1; fi
if [ -z "$NUM_NODES" ]; then echo "Error: Could not read 'num_nodes' from $CONFIG_FILE"; exit 1; fi
if [ -z "$GPUS_PER_NODE" ]; then echo "Error: Could not read 'gpus_per_node' from $CONFIG_FILE"; exit 1; fi

# Construct model config path
MODEL_CONFIG_PATH="src/configs/model/${MODEL_IDENTIFIER}.yaml"
if [ ! -f "$MODEL_CONFIG_PATH" ]; then
    echo "Error: Model config not found at $MODEL_CONFIG_PATH"
    exit 1
fi

# Read tensor_parallel_size from model config
TENSOR_PARALLEL_SIZE=$(yq -r '.model.tensor_parallel_size' "$MODEL_CONFIG_PATH")

if [ -z "$TENSOR_PARALLEL_SIZE" ] || [ "$TENSOR_PARALLEL_SIZE" == "null" ]; then
    echo "Error: Could not read 'model.tensor_parallel_size' from $MODEL_CONFIG_PATH"
    exit 1
fi

# --- Validation ---
if [ "$GPUS_PER_NODE" -lt "$TENSOR_PARALLEL_SIZE" ]; then
    echo "Error: gpus_per_node ($GPUS_PER_NODE) < tensor_parallel_size ($TENSOR_PARALLEL_SIZE)"
    exit 1
fi
if [ $((GPUS_PER_NODE % TENSOR_PARALLEL_SIZE)) -ne 0 ]; then
    echo "Error: gpus_per_node ($GPUS_PER_NODE) not divisible by tensor_parallel_size ($TENSOR_PARALLEL_SIZE)"
    exit 1
fi
# --- End Validation ---

# Calculate ntasks_per_node
NTASKS_PER_NODE=$((GPUS_PER_NODE / TENSOR_PARALLEL_SIZE))

echo "Configuration:"
echo "  Nodes: $NUM_NODES"
echo "  GPUs per Node: $GPUS_PER_NODE"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "  Tasks per Node: $NTASKS_PER_NODE"

# Submit Slurm job
SBATCH_CMD="sbatch \
  --nodes=${NUM_NODES} \
  --gres=gpu:nvidia_h100_80gb_hbm3:${GPUS_PER_NODE} \
  --ntasks-per-node=${NTASKS_PER_NODE} \
  scripts/slurm/generate_solutions.sh \"$CONFIG_FILE\" \"$TENSOR_PARALLEL_SIZE\""

echo "Running: $SBATCH_CMD"
GPU_JOBID=$(eval "$SBATCH_CMD" | awk '{print $4}')
echo "Submitted GPU job with ID: $GPU_JOBID"
