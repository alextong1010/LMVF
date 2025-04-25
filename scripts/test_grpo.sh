#!/bin/bash
#SBATCH --job-name=lmvf_grpo
#SBATCH --nodes=3
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH -c 4                               # 4 cores per task
#SBATCH -t 01-00:00:00
#SBATCH -o grpo_logs/output_%j.log
#SBATCH -e grpo_logs/error_%j.log
#SBATCH -p seas_gpu
#SBATCH --account=hankyang_lab
#SBATCH --mem=32GB

# Load necessary modules
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate lmvf

export HF_HOME=/n/netscratch/hankyang_lab/Lab/alex/.cache/huggingface

module load gcc/14.2.0-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01 


# Get the list of allocated nodes
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))

# Assign the first 4 nodes for training and the 5th node for vLLM
TRAIN_NODES="${NODELIST[@]:0:2}"  # Nodes 0, 1 for training
VLLM_NODE="${NODELIST[2]}"  # Node 2 for vLLM


# Convert TRAIN_NODES array to a comma-separated string
TRAIN_NODES_CSV=$(IFS=,; echo "${TRAIN_NODES[*]}")

# Run training on the first 2 nodes (Group 1)
srun --nodes=2 --ntasks=2 --nodelist="$TRAIN_NODES_CSV" accelerate launch \
     --config_file deepspeed_zero3.yaml \
     --num_processes 2 \
     --num_machines 2 \
     --main_process_ip ${NODELIST[0]} \
     --main_process_port 29500 \
     --machine_rank $SLURM_PROCID \
     --rdzv_backend c10d \
     test_train.py \
     --vllm_server_host $VLLM_NODE > grpo_logs/train_output.log 2>&1 &


# Run vLLM server on the 3rd node (Group 2)
srun --nodes=1 --ntasks=1 --nodelist="$VLLM_NODE" trl vllm-serve --model google/gemma-3-1b-it --tensor_parallel_size 1 &

wait