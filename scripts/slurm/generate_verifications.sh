#!/bin/bash
#SBATCH --job-name=lmvf_gen_verifications
#SBATCH -c 4                               # 4 cores per task
#SBATCH -t 00-10:00:00
#SBATCH -o logs/output_%j.log
#SBATCH -e logs/error_%j.log
#SBATCH -p seas_gpu
#SBATCH --account=hankyang_lab
#SBATCH --mem=32GB

# === Input Arguments ===
CONFIG_FILE="$1"
TENSOR_PARALLEL_SIZE="$2"

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: No config file path provided."
    exit 1
fi
if [ -z "$TENSOR_PARALLEL_SIZE" ]; then
    echo "Error: No tensor_parallel_size provided."
    exit 1
fi

# Load necessary modules
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate lmvf

export HF_HOME=/n/netscratch/hankyang_lab/Lab/alex/.cache/huggingface

module load gcc/14.2.0-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01 

# Export the variable so it's available to the srun sub-shell
export TENSOR_PARALLEL_SIZE

# Run verification script per task, assigning specific GPUs via CUDA_VISIBLE_DEVICES
srun bash -c 'echo "Running generate_verifications.py with SLURM_NTASKS=$SLURM_NTASKS and SLURM_PROCID=$SLURM_PROCID on GPUs $SLURM_LOCALID"; \
              if [ $TENSOR_PARALLEL_SIZE -gt 1 ]; then \
                  START_GPU=$((SLURM_PROCID * TENSOR_PARALLEL_SIZE)); \
                  GPU_IDS=$(seq -s, $START_GPU $((START_GPU + TENSOR_PARALLEL_SIZE - 1))); \
                  export CUDA_VISIBLE_DEVICES=$GPU_IDS; \
              else \
                  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; \
              fi; \
              python scripts/py/generate_verifications.py --config-path '"$CONFIG_FILE"' --num-tasks=$SLURM_NTASKS --task-id=$SLURM_PROCID'
