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
- [ ] Add support for tensor_parallel_size > 1, especially with regards to CUDA_VISIBLE_DEVICES
- [ ] Run evals 
- [ ] Subclass GRPO trainer to use Verifiers as Reward funcs
- [ ] Implement multi-agent training loop
- [ ] Expand documentation and usage examples

