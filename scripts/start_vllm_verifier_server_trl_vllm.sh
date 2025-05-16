#!/bin/bash
#SBATCH --job-name=verifier_cluster
#SBATCH --nodes=1                         # <─ number of verifier GPUs
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 00-02:00:00
#SBATCH -p kempner_h100
#SBATCH --account=kempner_sham_lab
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
# : "${MODEL_ID:=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
: "${VLLM_PORT:=8002}"           # all nodes use same port

# ── 1.  Get node list and prepare output file ────────────────────────────────
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
OUT_FILE="$SLURM_SUBMIT_DIR/verifier_hosts.txt"

mkdir -p verifier_logs
: > "$OUT_FILE"                  # truncate / create

# ── 2.  Launch one vLLM server per node ───────────────────────────────────────
for node in "${NODELIST[@]}"; do
    # record "hostname:port" for the Python script
    echo "${node}:${VLLM_PORT}" >> "$OUT_FILE"

    # start the server on that node   (background - the & at the end)
    srun --nodes=1 --ntasks=1 --nodelist="$node" \
         trl vllm-serve --model "$MODEL_ID"       \
                        --tensor_parallel_size 1  \
                        --host 0.0.0.0            \
                        --port "$VLLM_PORT"       \
         > "verifier_logs/${node}_${SLURM_JOB_ID}.log" 2>&1 &
done

# ── 3.  Keep the allocation alive until the servers exit ─────────────────────
wait
