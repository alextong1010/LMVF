#!/bin/bash

# Change to repo root
cd "$(dirname "$0")/.."

# --- Configuration ---
CONFIG_FILE="configs/test_eval_config.yaml" # Choose: configs/test_eval_config.yaml or configs/test_train_config.yaml
CONFIG_FILE=$(realpath "$CONFIG_FILE")
SHAREDIR=temp/lmvf_shared # Changed name slightly for clarity
# --- End Configuration ---

# Ensure yq is available
if ! command -v yq &> /dev/null
then
    echo "yq could not be found. Please install yq (e.g., conda install yq -c conda-forge) and ensure it's in your PATH."
    exit 1
fi

# Ensure shared directory exists
if [ ! -d "$SHAREDIR" ]; then
    mkdir -p "$SHAREDIR"
    echo "Created shared directory: $SHAREDIR"
else
    echo "Shared directory already exists: $SHAREDIR"
fi
# echo "Cleaning up old files in $SHAREDIR..."
# rm -f $SHAREDIR/server_host_*.txt $SHAREDIR/server_ready_*.txt

# Read values from YAML config using yq (with -r flag for raw output)
GPUS_PER_NODE=$(yq -r '.gpus_per_node' "$CONFIG_FILE")
NUM_NODES=$(yq -r '.num_nodes' "$CONFIG_FILE")
MODEL_IDENTIFIER=$(yq -r '.model' "$CONFIG_FILE")

# Read verifier model settings
VERIFIER_MODEL=$(yq -r '.verifier_model' "$CONFIG_FILE")
VERIFIER_GPUS_PER_NODE=$(yq -r '.verifier_model_gpus_per_node' "$CONFIG_FILE")
VERIFIER_NUM_NODES=$(yq -r '.verifier_model_num_nodes' "$CONFIG_FILE")

# Validate required config values
if [ -z "$MODEL_IDENTIFIER" ]; then echo "Error: Could not read 'model' identifier from $CONFIG_FILE"; exit 1; fi
if [ -z "$NUM_NODES" ]; then echo "Error: Could not read 'num_nodes' from $CONFIG_FILE"; exit 1; fi
if ! [[ "$NUM_NODES" =~ ^[1-9][0-9]*$ ]]; then echo "Error: 'num_nodes' must be a positive integer."; exit 1; fi
if [ -z "$GPUS_PER_NODE" ]; then echo "Error: Could not read 'gpus_per_node' from $CONFIG_FILE"; exit 1; fi
if ! [[ "$GPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then echo "Error: 'gpus_per_node' must be a positive integer."; exit 1; fi

# Check if verifier model is specified
if [ "$VERIFIER_MODEL" != "null" ]; then
    # Validate verifier model config values
    if [ -z "$VERIFIER_GPUS_PER_NODE" ] || [ -z "$VERIFIER_NUM_NODES" ]; then
        echo "Error: Verifier model GPU settings are not properly configured."
        exit 1
    fi

    # Construct path to verifier model-specific config
    VERIFIER_MODEL_CONFIG_PATH="src/configs/model/${VERIFIER_MODEL}.yaml"
    if [ ! -f "$VERIFIER_MODEL_CONFIG_PATH" ]; then
        echo "Error: Verifier model config file not found at $VERIFIER_MODEL_CONFIG_PATH"
        exit 1
    fi

    # Read tensor_parallel_size for verifier model
    VERIFIER_TENSOR_PARALLEL_SIZE=$(yq -r '.model.tensor_parallel_size' "$VERIFIER_MODEL_CONFIG_PATH")
    if [ -z "$VERIFIER_TENSOR_PARALLEL_SIZE" ] || [ "$VERIFIER_TENSOR_PARALLEL_SIZE" == "null" ]; then
        echo "Error: Could not read 'model.tensor_parallel_size' for verifier model from $VERIFIER_MODEL_CONFIG_PATH"
        exit 1
    fi

    echo "Using verifier model tensor_parallel_size: $VERIFIER_TENSOR_PARALLEL_SIZE from $VERIFIER_MODEL_CONFIG_PATH"

    # --- Validation: GPUs per Node vs Tensor Parallel Size ---
    if [ "$VERIFIER_GPUS_PER_NODE" -lt "$VERIFIER_TENSOR_PARALLEL_SIZE" ]; then
        echo "Error: verifier_model_gpus_per_node ($VERIFIER_GPUS_PER_NODE) in $CONFIG_FILE must be greater than or equal to verifier_model_tensor_parallel_size ($VERIFIER_TENSOR_PARALLEL_SIZE) in $VERIFIER_MODEL_CONFIG_PATH"
        exit 1
    fi
    if [ $((VERIFIER_GPUS_PER_NODE % VERIFIER_TENSOR_PARALLEL_SIZE)) -ne 0 ]; then
        echo "Error: verifier_model_gpus_per_node ($VERIFIER_GPUS_PER_NODE) must be divisible by verifier_model_tensor_parallel_size ($VERIFIER_TENSOR_PARALLEL_SIZE)."
        exit 1
    fi
    # --- End Validation ---

    # Calculate number of servers per node and total servers
    VERIFIER_SERVERS_PER_NODE=$((VERIFIER_GPUS_PER_NODE / VERIFIER_TENSOR_PARALLEL_SIZE))
    VERIFIER_TOTAL_SERVERS=$((VERIFIER_NUM_NODES * VERIFIER_SERVERS_PER_NODE))

    echo "Verifier Model Configuration:"
    echo "  Nodes: $VERIFIER_NUM_NODES"
    echo "  GPUs per Node: $VERIFIER_GPUS_PER_NODE"
    echo "  Tensor Parallel Size: $VERIFIER_TENSOR_PARALLEL_SIZE"
    echo "  Servers per Node: $VERIFIER_SERVERS_PER_NODE"
    echo "  Total Servers: $VERIFIER_TOTAL_SERVERS"

    echo "Submitting verifier model GPU job from $(pwd)"


    # Submit verifier model server job
    echo "Submitting verifier model server job with $VERIFIER_TOTAL_SERVERS tasks across $VERIFIER_NUM_NODES nodes..."
    SBATCH_VERIFIER_CMD="sbatch \
      --chdir=\"$(pwd)\" \
      --nodes=${VERIFIER_NUM_NODES} \
      --ntasks=${VERIFIER_TOTAL_SERVERS} \
      --gres=gpu:${VERIFIER_GPUS_PER_NODE} \
      --gpus-per-task=${VERIFIER_TENSOR_PARALLEL_SIZE} \
      scripts/slurm/gpu_server_job.sh \"$CONFIG_FILE\" \"$VERIFIER_TENSOR_PARALLEL_SIZE\" \"verifier\""

    echo "Running: $SBATCH_VERIFIER_CMD"
    VERIFIER_GPU_JOBID=$(eval "$SBATCH_VERIFIER_CMD" | awk '{print $4}')
    echo "Submitted verifier model GPU server job with ID: $VERIFIER_GPU_JOBID"
fi

# Construct path to model-specific config
MODEL_CONFIG_PATH="src/configs/model/${MODEL_IDENTIFIER}.yaml"
if [ ! -f "$MODEL_CONFIG_PATH" ]; then
    echo "Error: Model config file not found at $MODEL_CONFIG_PATH"
    exit 1
fi

# Read tensor_parallel_size from model config (with -r flag)
TENSOR_PARALLEL_SIZE=$(yq -r '.model.tensor_parallel_size' "$MODEL_CONFIG_PATH")
if [ -z "$TENSOR_PARALLEL_SIZE" ] || [ "$TENSOR_PARALLEL_SIZE" == "null" ]; then
    echo "Error: Could not read 'model.tensor_parallel_size' from $MODEL_CONFIG_PATH"
    exit 1
fi

echo "Using tensor_parallel_size: $TENSOR_PARALLEL_SIZE from $MODEL_CONFIG_PATH"

# --- Validation: GPUs per Node vs Tensor Parallel Size ---
if [ "$GPUS_PER_NODE" -lt "$TENSOR_PARALLEL_SIZE" ]; then
    echo "Error: gpus_per_node ($GPUS_PER_NODE) in $CONFIG_FILE must be greater than or equal to tensor_parallel_size ($TENSOR_PARALLEL_SIZE) in $MODEL_CONFIG_PATH"
    exit 1
fi
if [ $((GPUS_PER_NODE % TENSOR_PARALLEL_SIZE)) -ne 0 ]; then
    echo "Error: gpus_per_node ($GPUS_PER_NODE) must be divisible by tensor_parallel_size ($TENSOR_PARALLEL_SIZE)."
    exit 1
fi
# --- End Validation ---

# Calculate number of servers per node and total servers for the main model
SERVERS_PER_NODE=$((GPUS_PER_NODE / TENSOR_PARALLEL_SIZE))
TOTAL_SERVERS=$((NUM_NODES * SERVERS_PER_NODE))


echo "Configuration:"
echo "  Nodes: $NUM_NODES"
echo "  GPUs per Node: $GPUS_PER_NODE"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "  Servers per Node: $SERVERS_PER_NODE"
echo "  Total Servers: $TOTAL_SERVERS"

echo "Submitting GPU job from $(pwd)"


# Submit GPU server job(s)
# Request NUM_NODES nodes, each with GPUS_PER_NODE GPUs.
# Start TOTAL_SERVERS tasks in total across these nodes.
# Each task (server) requires TENSOR_PARALLEL_SIZE GPUs.
echo "Submitting server job with $TOTAL_SERVERS tasks across $NUM_NODES nodes..."
SBATCH_CMD="sbatch \
  --chdir=\"$(pwd)\" \
  --nodes=${NUM_NODES} \
  --ntasks=${TOTAL_SERVERS} \
  --gres=gpu:${GPUS_PER_NODE} \
  --gpus-per-task=${TENSOR_PARALLEL_SIZE} \
  scripts/slurm/gpu_server_job.sh \"$CONFIG_FILE\" \"$TENSOR_PARALLEL_SIZE\""

echo "Running: $SBATCH_CMD"
GPU_JOBID=$(eval "$SBATCH_CMD" | awk '{print $4}')

echo "Submitted GPU server job with ID: $GPU_JOBID using config $CONFIG_FILE"

# Initialize total servers including verifier
TOTAL_SERVERS_WITH_VERIFIER=$TOTAL_SERVERS

# Check if verifier model is specified
if [ "$VERIFIER_MODEL" != "null" ]; then
    # Add verifier total servers to the total
    TOTAL_SERVERS_WITH_VERIFIER=$((TOTAL_SERVERS + VERIFIER_TOTAL_SERVERS))
fi

echo "Total Servers (including verifier if applicable): $TOTAL_SERVERS_WITH_VERIFIER"

# Submit *one* client job, dependent on the server job finishing successfully
# Allocate CPUs and Memory based on the total number of servers
CLIENT_CPUS=$((TOTAL_SERVERS_WITH_VERIFIER > 0 ? TOTAL_SERVERS_WITH_VERIFIER : 1)) # At least 1 CPU
CLIENT_MEM_GB=$((TOTAL_SERVERS_WITH_VERIFIER * 4))                  # 4 GB per server
if [ "$CLIENT_MEM_GB" -lt 4 ]; then CLIENT_MEM_GB=4; fi # Minimum 4GB memory
echo "Submitting single client job with $CLIENT_CPUS CPUs and ${CLIENT_MEM_GB}G Memory..."
CLIENT_JOBID=$(sbatch \
    --chdir="$(pwd)" \
    --dependency=after:$GPU_JOBID \
    --ntasks=1 \
    --cpus-per-task=${CLIENT_CPUS} \
    --mem=${CLIENT_MEM_GB}G \
    scripts/slurm/client_job.sh "$CONFIG_FILE" "$TOTAL_SERVERS_WITH_VERIFIER" | awk '{print $4}') # Pass TOTAL_SERVERS_WITH_VERIFIER
echo "Submitted Client job with ID: $CLIENT_JOBID"

# Submit cleanup job to run after the *single* client job finishes
echo "Submitting cleanup job to remove $SHAREDIR after client job finishes..."
CLEANUP_JOBID=$(sbatch \
    --chdir="$(pwd)" \
    --dependency=afterany:$CLIENT_JOBID \
    --ntasks=1 \
    --job-name=cleanup_sharedir \
    --output=logs/slurm-%j.out \
    --wrap="rm -rf temp/ && echo 'Cleaned up shared directory: $SHAREDIR'" | awk '{print $4}')
echo "Submitted cleanup job with ID: $CLEANUP_JOBID"
