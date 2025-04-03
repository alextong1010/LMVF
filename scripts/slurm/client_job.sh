#!/bin/bash
#SBATCH --job-name=gemma-client
#SBATCH --output=slurm-client-%j.out
#SBATCH --account=hankyang_lab
#SBATCH --nodes=1
#SBATCH --partition=seas_compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -t 00-00:10
#SBATCH --mem=8GB


# Move to repo root
cd "$(dirname "$0")/../.."

# Shared paths
SHAREDIR=temp/gemma_test
READY_FILE=$SHAREDIR/server_ready.txt
HOST_FILE=$SHAREDIR/server_host.txt

# Load necessary modules
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate lmvf

# Wait for hostname
while [ ! -f $HOST_FILE ]; do
    echo "Waiting for server host info..."
    sleep 2
done

# Wait for readiness signal
while [ ! -f $READY_FILE ]; do
    echo "Waiting for server to be ready..."
    sleep 2
done

# Optionally wait for HTTP availability
SERVER_HOST=$(cat $HOST_FILE)
until curl -s http://$SERVER_HOST:8000/v1/models; do
    echo "Waiting for server to respond..."
    sleep 3
done

# Run client script (no need to pass server URL)
python src/eval/inference.py
