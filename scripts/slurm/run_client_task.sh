#!/bin/bash
set -e

# --- Get Config File and Total Servers from Arguments ---
if [ -z "$1" ]; then
    echo "[Client] Error: Config file path not provided as the first argument."
    exit 1
fi
CONFIG_FILE=$1

if [ -z "$2" ]; then
    echo "[Client] Error: Total number of servers not provided as the second argument."
    exit 1
fi
TOTAL_SERVERS=$2
# --- End Arguments ---

# Client task doesn't have a meaningful SLURM_PROCID in the same way as before (it's always 0 for a single task job)
echo "[Client] Running on $(hostname) | PWD=$(pwd)"
echo "[Client] Expecting $TOTAL_SERVERS server(s)."

# Shared directory
SHAREDIR=temp/lmvf_shared
HOST_FILE_COMMON=$SHAREDIR/server_hosts.txt
VERIFIER_HOST_FILE=$SHAREDIR/verifier_server_hosts.txt

# Load necessary modules
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate lmvf

# --- Wait for all servers to report host info and signal readiness ---
echo "[Client] Waiting for $TOTAL_SERVERS hosts in $HOST_FILE_COMMON and for $TOTAL_SERVERS ready signals (server_ready_*.txt)..."

MAX_WAIT_ITERATIONS=90 # Increased iterations (e.g., 90 * 10s = 15 minutes)
WAIT_COUNT=0
READY_COUNT=0

# Function to count all ready files
count_ready_files() {
    local ready_files_count
    ready_files_count=$(ls "$SHAREDIR"/server_ready_*.txt 2>/dev/null | wc -l)
    echo "$ready_files_count"
}

# Loop until the host file exists and has the right number of lines, AND all ready files exist
while [ $WAIT_COUNT -lt $MAX_WAIT_ITERATIONS ]; do
    # Check host file count
    if [ -f "$HOST_FILE_COMMON" ]; then
        HOST_COUNT=$(wc -l < "$HOST_FILE_COMMON")
    else
        HOST_COUNT=0
    fi

    # Check verifier host file count if verifier model is used
    if [ -f "$VERIFIER_HOST_FILE" ]; then
        VERIFIER_HOST_COUNT=$(wc -l < "$VERIFIER_HOST_FILE")
    else
        VERIFIER_HOST_COUNT=0
    fi

    # Check ready file count
    READY_COUNT=$(count_ready_files)

    # Check if conditions met (HOST_COUNT + VERIFIER_HOST_COUNT = TOTAL_SERVERS_WITH_VERIFIER)
    TOTAL_COUNT_WITH_VERIFIER=$((HOST_COUNT + VERIFIER_HOST_COUNT))
    if [ "$TOTAL_COUNT_WITH_VERIFIER" -eq "$TOTAL_SERVERS" ] && [ "$READY_COUNT" -ge "$TOTAL_SERVERS" ]; then
        echo "[Client] Found $TOTAL_COUNT_WITH_VERIFIER/$TOTAL_SERVERS hosts and $READY_COUNT/$TOTAL_SERVERS ready signals. Conditions met."
        break # Exit loop successfully
    fi

    # Print status and sleep
    echo "[Client] Waiting... ($WAIT_COUNT/$MAX_WAIT_ITERATIONS). Found $TOTAL_COUNT_WITH_VERIFIER/$TOTAL_SERVERS hosts. Found $READY_COUNT/$TOTAL_SERVERS ready signals."
    sleep 10
    WAIT_COUNT=$((WAIT_COUNT + 1))
done

# Final check after loop
if [ "$TOTAL_COUNT_WITH_VERIFIER" -lt "$TOTAL_SERVERS" ]; then
    echo "[Client] Error: Timed out. Found only $TOTAL_COUNT_WITH_VERIFIER/$TOTAL_SERVERS hosts in $HOST_FILE_COMMON."
    ls -l "$SHAREDIR" # List files for debugging
    exit 1
fi
if [ "$READY_COUNT" -lt "$TOTAL_SERVERS" ]; then
    echo "[Client] Error: Timed out. Found only $READY_COUNT/$TOTAL_SERVERS ready signals (server_ready_*.txt)."
    ls -l "$SHAREDIR"
    exit 1
fi

echo "[Client] All $TOTAL_SERVERS servers reported host info and signaled ready."

# --- HTTP Health Check for All Servers ---
echo "[Client] Performing HTTP readiness check for all servers listed in $HOST_FILE_COMMON and $VERIFIER_HOST_FILE..."

# Combine host files for health check
cat "$HOST_FILE_COMMON" "$VERIFIER_HOST_FILE" > "$SHAREDIR/all_server_hosts.txt"

MAX_RETRIES=50
RETRY_DELAY=5  # seconds

SERVER_INDEX=0
while IFS= read -r SERVER_HOST_PORT; do
    HEALTH_CHECK_URL="http://${SERVER_HOST_PORT}/v1/models"  # Use /health if your server supports it

    RETRY_COUNT=0
    until curl --max-time 5 -f -s "${HEALTH_CHECK_URL}" > /dev/null || [ $RETRY_COUNT -ge $MAX_RETRIES ]; do
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo "[Client] Waiting for server $SERVER_HOST_PORT to respond at ${HEALTH_CHECK_URL} (Attempt $RETRY_COUNT/$MAX_RETRIES)..."
        sleep $RETRY_DELAY
    done

    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "[Client] Error: Server $SERVER_HOST_PORT did not become ready after $MAX_RETRIES attempts."
        echo "[Client] Last curl attempt details:"
        curl -v "${HEALTH_CHECK_URL}"
        exit 1
    fi

    echo "[Client] Server $SERVER_HOST_PORT is HTTP-ready."
    SERVER_INDEX=$((SERVER_INDEX + 1))
done < "$SHAREDIR/all_server_hosts.txt"

echo "[Client] All servers passed HTTP readiness check."
# --- End HTTP Health Check ---

echo "[Client] Proceeding to run Python client."
# --- End Wait ---

# Run the central client inference script
# Pass the total number of servers it needs to coordinate with
echo "[Client] Launching Python inference script (run_inference_test.py)..."
python src/eval/run_inference_test.py \
    --total-servers "$TOTAL_SERVERS" \
    --config-file "$CONFIG_FILE" \
    --shared-dir "$SHAREDIR"

echo "[Client] Client script finished."