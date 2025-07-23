# LMVF
Learning with Multi-agent Verifier Feedback

Alex Tong, Yilun Du

# Try Multi-Agent Verification

## Setup

We recommend using a conda environment with Python 3.10.
1. conda create -n lmvf python=3.10 
2. pip install trl
3. pip install trl[vllm]
4. pip install vllm==0.8.1 (currently on '0.8.5.post1')
5. pip install tokenizer==0.21.0 (will say its incompatible, but this is what is needed to get gemma3 working on 0.8.5.post1)
5. conda install -c conda-forge yq
6. conda install -c conda-forge jq

Note: 0.8.5.post1 is recommended for every model except gemma3. If you want to use gemma3 on 0.8.5.post1, you need to downgrade tokenizer to 0.21.0. Otherwise, just ignore gemma3. Even with this, gemma3 has slow throughput. 
if you want faster gemma3 throughput at the cost of slower throughputs in other models, use vllm == 0.8.1.

WIP:
add custom flag + options for bon-mav, i.e. selecting which generations to use (to run more extensive tests), as well as option to change num_verifiers and which specific ones