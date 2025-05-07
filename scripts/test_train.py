# SPDX-License-Identifier: Apache-2.0
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# train_grpo.py
from src.utils.loading_utils import load_and_validate_config, parse_vllm_args, parse_initial_arguments
from src.utils.dataset_utils import load_train_dataset
from trl import GRPOConfig, GRPOTrainer
from src.prompts.gen_prompts import get_gen_prompt
from transformers import Gemma3ForCausalLM
from src.utils.reward_verifiers import LLMVerifier


def main(config, model_config, verifier_model_config, strict_verifier_model_config, args, verifier_args, strict_verifier_args, task_id, num_tasks, verbose):
    dataset = load_train_dataset(config, task_id, num_tasks)
    # dataset = load_dataset("trl-lib/tldr", split="train")
    model_name = "google/gemma-3-1b-it"

    def to_chat_prompt(problem):
        return {
            "prompt": [{"role": "user", "content":  [{"type": "text", "text": get_gen_prompt(config['dataset'], problem['problem'])}]}]
        }
    dataset = dataset.map(to_chat_prompt)

    with open("verifier_hosts_13005870.txt") as f:
        hosts_ports = [tuple(line.strip().split(":")) for line in f]

    # hosts_ports now looks like [("ip-10-0-0-5", "8000"), ("ip-10-0-0-6", "8000"), ...]
    verifier_reward = LLMVerifier(
            [(h, int(p)) for h, p in hosts_ports],
            model_name, config, verifier_model_config, strict_verifier_model_config, dataset)

    # Define the reward function, which rewards completions that are close to 20 characters
    def reward_len(completions, **kwargs):
        return [-abs(20 - len(completion)) for completion in completions]

    # Load Gemma 3 model with the recommended *eager* attention implementation
    model = Gemma3ForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",  # avoid SDPA for Gemma 3
        torch_dtype="bfloat16",        # keep dtype consistent with bf16 training
    )

    training_args = GRPOConfig(output_dir="/n/netscratch/hankyang_lab/Lab/alex/grpo_training_output/Gemma-3-1B-GRPO", logging_steps=10)

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs  = [reward_len, verifier_reward],
        train_dataset=dataset,
    )
    trainer.train()

if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(__file__)) # Adjust if structure changes
    config_args, remaining_vllm_args = parse_initial_arguments(repo_root)

    task_id = config_args.task_id
    num_tasks = config_args.num_tasks
    verbose = config_args.verbose

    # Load configuration using the parsed path
    config, model_config, model_path, verifier_model_config, verifier_model_path, strict_verifier_model_config, strict_verifier_model_path = load_and_validate_config(config_args, repo_root)
    # Parse the generator arguments using the modified parse_vllm_args
    print("Parsing generator arguments...")
    args = parse_vllm_args(model_path, remaining_vllm_args) # Pass remaining args
    print("Generator arguments parsed.")

    verifier_args = None
    if verifier_model_path:
        print("Parsing verifier arguments...")
        verifier_args = parse_vllm_args(verifier_model_path, remaining_vllm_args)
        print("Verifier arguments parsed.")

        if verbose:
            print(f"Generator Args: {args}")
            print(f"Verifier Args: {verifier_args}")
    else:
        print("No separate verifier model specified, verifier_args will be None.")

    strict_verifier_args = None
    if strict_verifier_model_path:
        print("Parsing strict verifier arguments...")
        strict_verifier_args = parse_vllm_args(strict_verifier_model_path, remaining_vllm_args)
        print("Strict Verifier arguments parsed.")
    main(config, model_config, verifier_model_config, strict_verifier_model_config, args, verifier_args, strict_verifier_args, task_id, num_tasks, verbose)