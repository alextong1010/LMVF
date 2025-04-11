#!/bin/bash
#SBATCH --job-name=gemma-client
#SBATCH --output=logs/slurm-client-%j.out # Single client, no task ID needed in filename
#SBATCH --account=hankyang_lab
#SBATCH --nodes=1
#SBATCH --partition=seas_compute
#SBATCH -t 00-00:30
#
# Slurm directives controlled by submit_jobs.sh:
# --ntasks=1
# --cpus-per-task=C (where C = total number of servers)
# --mem=M (where M = total number of servers * 4G, min 4G)
# --dependency=after:GPU_JOBID

# Use srun with --ntasks=1 since the job itself requests 1 task.
# Pass all arguments received by this script ($@) to the next script.
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK bash scripts/slurm/run_client_task.sh "$@"

