#!/bin/bash
#SBATCH --job-name=gemma-client
#SBATCH --output=slurm-client-%j-%t.out # Added task ID to output filename
#SBATCH --account=hankyang_lab
#SBATCH --nodes=1
#SBATCH --partition=seas_compute
#SBATCH --cpus-per-task=2
#SBATCH -t 00-00:30
#SBATCH --mem=8GB
#
# Slurm directives controlled by submit_jobs.sh:
# --ntasks=M (where M = total number of servers)

srun --ntasks=$SLURM_NTASKS bash scripts/slurm/run_client_task.sh "$@"

