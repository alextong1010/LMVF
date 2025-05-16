#!/bin/bash
#SBATCH --job-name=verifier_cluster
#SBATCH --nodes=1                         # <─ number of verifier GPUs
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 01-00:00:00
#SBATCH -p gpu
#SBATCH --mem=32GB
#SBATCH -o verifier_logs/out_%j.log
#SBATCH -e verifier_logs/err_%j.log

# ── 0.  Environment ───────────────────────────────────────────────────────────
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate lmvf
module load gcc/14.2.0-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01
export HF_HOME=/n/netscratch/hankyang_lab/Lab/alex/.cache/huggingface

# If you want a different verifier model, export MODEL_ID before sbatch:
: "${MODEL_ID:=Qwen/Qwen2.5-3B-Instruct}"
: "${VLLM_PORT:=8002}"           # all nodes use same port

# ── 1.  Get node list and prepare output file ────────────────────────────────
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
OUT_FILE="$SLURM_SUBMIT_DIR/verifier_hosts.txt"

mkdir -p verifier_logs
: > "$OUT_FILE"                  # truncate / create

echo "${NODELIST[0]}:${VLLM_PORT}" >> "$OUT_FILE"

vllm serve "$MODEL_ID" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 & # Bind to all interfaces within the node

    > "verifier_logs/${NODELIST[0]}_${SLURM_JOB_ID}.log" 2>&1 &

# ── 3.  Keep the allocation alive until the servers exit ─────────────────────
wait
