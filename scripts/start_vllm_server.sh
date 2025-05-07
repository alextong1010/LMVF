#!/bin/bash
#SBATCH --job-name=lmvf_grpo
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -c 4                               # 4 cores per task
#SBATCH -t 02:00:00
#SBATCH -o vllm_logs/output_%j.log
#SBATCH -e vllm_logs/error_%j.log
#SBATCH -p gpu
#SBATCH --account=hankyang_lab
#SBATCH --mem=32GB

# Load necessary modules
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate lmvf

export HF_HOME=/n/netscratch/hankyang_lab/Lab/alex/.cache/huggingface

module load gcc/14.2.0-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01 

export VLLM_WORKER_MULTIPROC_METHOD=spawn

# trl vllm-serve --model Qwen/Qwen2.5-0.5B-Instruct

# # Get the list of allocated nodes
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))

VLLM_NODE="${NODELIST[0]}"  # Node 0 for vLLM

echo "VLLM_NODE: $VLLM_NODE"

# # Run vLLM server on the 3rd node (Group 2)
srun --nodes=1 --ntasks=1 --nodelist="$VLLM_NODE" trl vllm-serve --model Qwen/Qwen2.5-0.5B-Instruct --tensor_parallel_size 1 &

wait