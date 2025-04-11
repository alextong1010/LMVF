#!/bin/bash
set -e

# --- Get Config File and TP Size from Arguments ---
if [ -z "$1" ]; then
    echo "[Task $SLURM_PROCID] Error: Config file path not provided as the first argument."
    exit 1
fi
CONFIG_FILE=$1

if [ -z "$2" ]; then
    echo "[Task $SLURM_PROCID] Error: Tensor parallel size not provided as the second argument."
    exit 1
fi
TENSOR_PARALLEL_SIZE=$2

# Check if this is a verifier model task
IS_VERIFIER_MODEL=false
VERIFIER_TASK_OFFSET=0 # Default offset is 0 for generator tasks
if [ "$3" == "verifier" ]; then
    IS_VERIFIER_MODEL=true
    VERIFIER_TASK_OFFSET=20 # Fixed offset of 20 for verifier tasks
    echo "[Task $SLURM_PROCID] Running verifier model server task."
fi
# --- End Arguments ---
TASK_ID=$((SLURM_PROCID + VERIFIER_TASK_OFFSET))
echo "[Task $TASK_ID] Running on $(hostname) | CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES | PWD=$(pwd)"


# Set shared directory
SHAREDIR=temp/lmvf_shared

# Determine the correct host file based on whether this is a verifier model task
if [ "$IS_VERIFIER_MODEL" == true ]; then
    HOST_FILE_COMMON=$SHAREDIR/verifier_server_hosts.txt
else
    HOST_FILE_COMMON=$SHAREDIR/server_hosts.txt
fi

# Task-specific identifiers - SLURM_PROCID is unique across all tasks (0 to TOTAL_SERVERS - 1)
if [ -z "$TASK_ID" ]; then
  echo "Warning: SLURM_PROCID not set, defaulting TASK_ID to 0."
  TASK_ID=0
fi
# Base port + task ID ensures unique port per server instance
PORT=$((8000 + TASK_ID))

READY_FILE=$SHAREDIR/server_ready_$TASK_ID.txt

echo "[Task $TASK_ID] Starting on $(hostname). Will use port $PORT. TP Size: $TENSOR_PARALLEL_SIZE."

# Load env
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate lmvf # Make sure yq is installed in this environment: conda install yq -c conda-forge

# Check if yq is available within the environment
if ! command -v yq &> /dev/null
then
    echo "[Task $TASK_ID] Error: yq command not found in conda environment. Please install it."
    exit 1
fi


module load cudnn/9.5.1.17_cuda12-fasrc01 # Ensure compatible CUDA/CuDNN
module load cuda/12.4.1-fasrc01

# --- Read Model Identifier from Main Config ---
MODEL_IDENTIFIER=$(yq -r '.model' "$CONFIG_FILE")
if [ "$IS_VERIFIER_MODEL" == true ]; then
    MODEL_IDENTIFIER=$(yq -r '.verifier_model' "$CONFIG_FILE")
fi
if [ -z "$MODEL_IDENTIFIER" ]; then
    echo "[Task $TASK_ID] Error: Could not read 'model' identifier from $CONFIG_FILE"
    exit 1
fi
echo "[Task $TASK_ID] Using model identifier: $MODEL_IDENTIFIER from config $CONFIG_FILE"
# --- End Read Model Identifier ---

# --- Construct Path to Model-Specific Config ---
MODEL_CONFIG_PATH="src/configs/model/${MODEL_IDENTIFIER}.yaml"
if [ ! -f "$MODEL_CONFIG_PATH" ]; then
    echo "[Task $TASK_ID] Error: Model config file not found at $MODEL_CONFIG_PATH"
    exit 1
fi
echo "[Task $TASK_ID] Reading model details from: $MODEL_CONFIG_PATH"
# --- End Construct Path ---

# --- Read Model Path from Model-Specific Config ---
MODEL_PATH=$(yq -r '.model.path' "$MODEL_CONFIG_PATH")
if [ -z "$MODEL_PATH" ] || [ "$MODEL_PATH" == "null" ]; then
    echo "[Task $TASK_ID] Error: Could not read 'model.path' from $MODEL_CONFIG_PATH"
    exit 1
fi
echo "[Task $TASK_ID] Using model path: $MODEL_PATH"
# --- End Read Model Path ---

# Save host info (hostname:port) by appending to the appropriate host file
HOSTNAME=$(hostname)
HOST_PORT_INFO="${HOSTNAME}:${PORT}"
echo "$HOST_PORT_INFO" >> "$HOST_FILE_COMMON" # Append to the correct host file
echo "[Task $TASK_ID] Host information appended to $HOST_FILE_COMMON: $HOST_PORT_INFO"

# Start vLLM server on separate port per task using model path and TP size
# Slurm's --gpus-per-task handles CUDA_VISIBLE_DEVICES
echo "[Task $TASK_ID] Starting vLLM server for model $MODEL_PATH with TP size $TENSOR_PARALLEL_SIZE on port $PORT..."
vllm serve "$MODEL_PATH" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --host 0.0.0.0 & # Bind to all interfaces within the node

VLLM_PID=$!

# Wait for it to initialize (implement a more robust health check)
# Increased sleep significantly as multi-GPU/larger models take longer
echo "[Task $TASK_ID] Waiting for vLLM server PID $VLLM_PID to initialize (sleeping 60s)..."
sleep 60

# Basic check if process is still running
if ! ps -p $VLLM_PID > /dev/null; then
   echo "[Task $TASK_ID] Error: vLLM server process $VLLM_PID exited prematurely."
   # Optionally, try to cat logs if vllm logs somewhere specific
   exit 1
fi

# Signal that this server is ready by creating its specific ready file
touch "$READY_FILE"
echo "[Task $TASK_ID] Server is ready on port $PORT with model $MODEL_PATH (TP=$TENSOR_PARALLEL_SIZE)"

wait $VLLM_PID # Wait specifically for the vllm process to finish
echo "[Task $TASK_ID] vLLM server process $VLLM_PID finished."