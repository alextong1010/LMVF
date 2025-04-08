#!/bin/bash
#SBATCH --job-name=gemma-server
#SBATCH --output=slurm-gpu-%j-%t.out 
#SBATCH --account=hankyang_lab
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=00-00:10
#SBATCH --partition=seas_gpu
#
# Slurm directives controlled by submit_jobs.sh:
# --nodes=N
# --ntasks=M (where M = N * servers_per_node)
# --gres=gpu:G (where G = gpus_per_node)
# --gpus-per-task=T (where T = tensor_parallel_size)

srun --ntasks=$SLURM_NTASKS bash scripts/slurm/run_server_task.sh "$@"

