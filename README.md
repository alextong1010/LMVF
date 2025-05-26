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
3. Gemma 3 doesnt work with vllm 0.8.5 post 1, which is what trl[vllm] installs.

Note: 
1. train Number theory MATH 7115 and 7117 have empty \\boxed{} answers. Fix. Add 0 for both. 
2. algebra/25040 and algebra/24014 MATH trainhave missing {}. add them


To Run:
python run_eval.py --config-path configs/eval_config.yaml (1 GPU)

or 

./slurm_eval.sh (For data parallelism)

To Run (Train):
python train.py --config-path configs/train_config.yaml

To confirm/do:
1. Change grpo_config max_prompt_length to None
2. Change grpo_config max_completion_length to max_new_tokens from model_config
(Done) 3. Change grpo_config temperature, top_k, top_p, min_p for each model
4. Modify grpo_config based on the model configs files (temp, top p, top k, min p)
5. Check verifier_args and strict_verifier_args, what are they? And do any args need to be changed?
6. Check if models other than qwen2.5 0.5b instruct need to use to_chat_prompt_v3
7. Debug for slurm, i.e. what is the slurm training script looking like with vllm colocate?
8. Remove colocate and use servers instead
9. For models, figure out which prompting version is utilized
10. make training and eval more efficient
11. Modify run_initial_verification_batch to parse both borda and bonmav to make it more efficient, if borda works

Note:
In general, only instruction-tuned models have a chat template. Base models may perform poorly as they are not trained to respond to the chat conversation.
It fails (    raise RuntimeError(f"NCCL error: {error_str}")
RuntimeError: NCCL error: unhandled cuda error (run with NCCL_DEBUG=INFO for details) ) if the vllm server and the train.py is on the same node (? I think)

## Work in Progress
- [Done ] Allow for custom model switching just by switching the config.yaml files
- [ Done] Test out dynamic gpu allocation given modifiable yaml configs
- [ Done, just test it] Add support for tensor_parallel_size > 1, especially with regards to CUDA_VISIBLE_DEVICES
- [ Done ] Run evals 
- [ Done ] create separate vllm node and just add api calls to the node as the verifier function.
- [ Done] Test num gen = 4, test eager implementation, test calling vllm server
- [ ] CHange the run_eval.py to eval.py after code finishes running
- [ ] Implement multi-agent training loop
- [ ] Expand documentation and usage examples

