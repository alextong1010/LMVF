#!/bin/bash
set -e

# --- Get Config File from Argument ---
if [ -z "$1" ]; then
    # Note: SLURM_PROCID might not be set if run outside sbatch or before job starts fully
    echo "[Client Task $SLURM_PROCID] Error: Config file path not provided as the first argument."
    exit 1
fi
CONFIG_FILE=$1
# --- End Config File ---

# Move to repo root
echo "[Task $TASK_ID] Working directory: $(pwd)"


# Shared directory
SHAREDIR=temp/gemma_test
# Get task ID and total number of tasks from SLURM environment
TASK_ID=$SLURM_PROCID
NUM_TASKS=$SLURM_NTASKS # Total number of client tasks == total number of server tasks
if [ -z "$TASK_ID" ]; then
  echo "Warning: SLURM_PROCID not set, defaulting TASK_ID to 0."
  TASK_ID=0
fi
if [ -z "$NUM_TASKS" ]; then
  echo "Warning: SLURM_NTASKS not set, defaulting NUM_TASKS to 1."
  NUM_TASKS=1
fi
echo "[Client Task $TASK_ID/$NUM_TASKS] Starting."


# Set file paths using TASK_ID (corresponds to server task ID)
HOST_FILE=$SHAREDIR/server_host_${TASK_ID}.txt
READY_FILE=$SHAREDIR/server_ready_${TASK_ID}.txt

# Load necessary modules
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate lmvf # Ensure run_inference.py dependencies are here

# Wait for hostname
while [ ! -f $HOST_FILE ]; do
    echo "Waiting for server host info ($HOST_FILE)..."
    sleep 5
done
echo "[Client Task $TASK_ID] Found host file $HOST_FILE."

# Wait for readiness signal
while [ ! -f $READY_FILE ]; do
    echo "Waiting for server to be ready ($READY_FILE)..."
    sleep 5
done
echo "[Client Task $TASK_ID] Found ready file $READY_FILE."


# Extract server address and port for this specific task
SERVER_HOST_PORT=$(cat "$HOST_FILE")
if [ -z "$SERVER_HOST_PORT" ]; then
    echo "[Client Task $TASK_ID] Error: Host file $HOST_FILE is empty."
    exit 1
fi
echo "[Client Task $TASK_ID] Server endpoint: $SERVER_HOST_PORT"

# Wait for HTTP readiness of the specific server
echo "[Client Task $TASK_ID] Checking HTTP readiness for $SERVER_HOST_PORT..."
# Adjust endpoint if needed, /v1/models is specific to OpenAI API compatibility layer
HEALTH_CHECK_URL="http://${SERVER_HOST_PORT}/health" # VLLM default health endpoint
# Fallback to models endpoint if /health isn't enabled or standard
# HEALTH_CHECK_URL="http://${SERVER_HOST_PORT}/v1/models"

MAX_RETRIES=50
RETRY_COUNT=0

until curl --max-time 5 -f -s "${HEALTH_CHECK_URL}" > /dev/null || [ $RETRY_COUNT -ge $MAX_RETRIES ]; do
    RETRY_COUNT=$((RETRY_COUNT+1))
    echo "[Client Task $TASK_ID] Waiting for server $SERVER_HOST_PORT to respond at ${HEALTH_CHECK_URL} (Attempt $RETRY_COUNT/$MAX_RETRIES)..."
    sleep 10 # Increased sleep
done

if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
  echo "[Client Task $TASK_ID] Error: Server $SERVER_HOST_PORT did not become ready after $MAX_RETRIES attempts."
  # Attempt to get more info if possible
  echo "[Client Task $TASK_ID] Last curl attempt details:"
  curl -v "${HEALTH_CHECK_URL}"
  exit 1
fi

echo "[Client Task $TASK_ID] Server $SERVER_HOST_PORT is ready. Running inference with config $CONFIG_FILE..."


# Run client inference, passing the config file path and task info
# Pass SHAREDIR to python script so it knows where to find host files (already done in previous step)
python src/eval/run_inference.py \
    --task-id "$TASK_ID" \
    --num-tasks "$NUM_TASKS" \
    --config-file "$CONFIG_FILE" \
    --shared-dir "$SHAREDIR" # Pass shared dir explicitly

echo "[Client Task $TASK_ID] Client inference finished."