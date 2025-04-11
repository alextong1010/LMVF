#!/bin/bash
#SBATCH --job-name=gemma-server
#SBATCH --output=logs/slurm-gpu-%j-%t.out 
#SBATCH --account=hankyang_lab
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=00-00:30
#SBATCH --partition=seas_gpu
#
# Slurm directives controlled by submit_jobs.sh:
# --nodes=N
# --ntasks=M (where M = N * servers_per_node)
# --gres=gpu:G (where G = gpus_per_node)
# --gpus-per-task=T (where T = tensor_parallel_size)

# Determine if this is a verifier model task
if [ "$3" == "verifier" ]; then
    echo "[Task $SLURM_PROCID] Running verifier model server task."
    # Additional logic for verifier model if needed
fi

srun --ntasks=$SLURM_NTASKS bash scripts/slurm/run_server_task.sh "$@"

