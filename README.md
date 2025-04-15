# LMVF
Learning with Multi-Agent Verifier Feedback

Alex Tong, Yilun Du

# Try Multi-Agent Verification

## Setup

We recommend using a conda environment with Python 3.10.
1. conda create -n lmvf python=3.10 
2. pip install trl
3. pip install trl[vllm]
4. pip install vllm
5. conda install -c conda-forge yq
6. conda install -c conda-forge jq


Common Errors
1. ImportError: libnccl.so.2: cannot open shared object file: No such file or directory
Try: conda install -c nvidia nccl

To Run:
python run_eval.py --config-path configs/eval_config.yaml

or 

./slurm_eval.sh (For data parallelism)

Note:
In general, only instruction-tuned models have a chat template. Base models may perform poorly as they are not trained to respond to the chat conversation.

## Work in Progress
- [Done ] Allow for custom model switching just by switching the config.yaml files
- [ Done] Test out dynamic gpu allocation given modifiable yaml configs
- [ Done] Spin up servers and test tensor_parallel_sizes for different models (maxs out at number of gpus available per node so 4. i.e. you can use at max 4 gpus per server) (Technically vllm allows for multi-node serving, but I'm too lazy and I don't need large models sooooo, not implemented!)
- [ Done ] Fix the bug
- [ Done ] Write the code to have generator_server_hosts and verifier_server_hosts (same hosts if theyre the same but still two different files)
- [ Done ] Modify run_server_task.sh and submit_jobs.sh to also run verifier servers
- [ ] Add support such that once the client script and cleanup finishes runnning, close the gpu servers
- [ ] rewrite the inference code and test it so that it supports one client and multiple servers and uses asyncio to do distributed inference with generators and verifiers efficiently without having to wait for one to finish
- [ ] Move everything from Huggingface to VLLM
- [ ] Write multinode+multigpu code for vllm
- [ ] Add support for different verifiers than generators - write the code to spin up multiple servers for verifiers and generators
- [ ] Rewrite Eval Script for pass@k + MAV
- [ ] Run evals 
- [ ] Subclass GRPO trainer to use Verifiers as Reward funcs
- [ ] Implement multi-agent training loop
- [ ] Expand documentation and usage examples

