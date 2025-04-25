# LMVF
Learning with Multi-Agent Verifier Feedback

Alex Tong, Yilun Du

# Try Multi-Agent Verification

## Setup

We recommend using a conda environment with Python 3.10.
1. conda create -n lmvf python=3.10 
2. pip install trl
3. pip install trl[vllm]
4. pip install vllm==0.8.1
5. pip instal 
5. conda install -c conda-forge yq
6. conda install -c conda-forge jq


Common Errors
1. ImportError: libnccl.so.2: cannot open shared object file: No such file or directory
Try: conda install -c nvidia nccl
2. Not sure why, but vllm 0.8.3 and vllm 0.8.4 triples the generation time and 10x the verification time of when using vllm 0.8.1


To Run:
python run_eval.py --config-path configs/eval_config.yaml (1 GPU)

or 

./slurm_eval.sh (For data parallelism)

Note:
In general, only instruction-tuned models have a chat template. Base models may perform poorly as they are not trained to respond to the chat conversation.

## Work in Progress
- [Done ] Allow for custom model switching just by switching the config.yaml files
- [ Done] Test out dynamic gpu allocation given modifiable yaml configs
- [ Done, just test it] Add support for tensor_parallel_size > 1, especially with regards to CUDA_VISIBLE_DEVICES
- [ ] Run evals 
- [ ] create separate vllm node and just add api calls to the node as the verifier function.
- [ ] Test num gen = 4, test eager implementation, test calling vllm server
- [ ] Implement multi-agent training loop
- [ ] Expand documentation and usage examples

