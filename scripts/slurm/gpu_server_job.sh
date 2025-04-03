#!/bin/bash
#SBATCH --job-name=gemma-server
#SBATCH --output=slurm-gpu-%j.out
#SBATCH --account=hankyang_lab
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -t 00-00:10
#SBATCH --partition=seas_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB

# Move to repo root
cd "$(dirname "$0")/../.."

# Shared file paths
SHAREDIR=../../temp/gemma_test
READY_FILE=$SHAREDIR/server_ready.txt
HOST_FILE=$SHAREDIR/server_host.txt

mkdir -p $SHAREDIR

# Load necessary modules
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh

# Save hostname so client knows where the server is
echo $(hostname) > $HOST_FILE

module load cudnn/9.5.1.17_cuda12-fasrc01 
module load cuda/12.4.1-fasrc01


conda activate lmvf

# Start vLLM server in background
vllm serve google/gemma-3-1b-it --port 8000 &

# Wait for server to start up (or replace with health check)
sleep 20

# Signal readiness
touch $READY_FILE
echo "Server is ready."

wait
