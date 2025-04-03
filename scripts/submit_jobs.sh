#!/bin/bash

# Change to repo root
cd "$(dirname "$0")/.."

GPU_JOBID=$(sbatch scripts/slurm/gpu_server_job.sh | awk '{print $4}')
echo "Submitted GPU server job with ID: $GPU_JOBID"

sbatch --dependency=after:$GPU_JOBID scripts/slurm/client_job.sh
