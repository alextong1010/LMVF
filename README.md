# LMVF
Learning with Multi-Agent Verifier Feedback

Alex Tong, Yilun Du

# Try Multi-Agent Verification

## Setup

We recommend using a conda environment with Python 3.10.
1. conda create -n lmvf python=3.10 
2. pip install trl
3. pip install trl[vllm]
4. 


Common Errors
1. ImportError: libnccl.so.2: cannot open shared object file: No such file or directory
Try: conda install -c nvidia nccl


## Work in Progress

- [ ] Test out dynamic gpu allocation given modifiable yaml configs
- [ ] Spin up servers and test tensor_parallel_sizes for different models
- [ ] Move everything from Huggingface to VLLM
- [ ] Write multinode+multigpu code for vllm
- [ ] Rewrite Eval Script for pass@k + MAV
- [ ] Run evals 
- [ ] Subclass GRPO trainer to use Verifiers as Reward funcs
- [ ] Implement multi-agent training loop
- [ ] Expand documentation and usage examples


