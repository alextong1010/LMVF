# SPDX-License-Identifier: Apache-2.0
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# train_grpo.py
from src.utils.loading_utils import load_and_validate_config, parse_vllm_args, parse_initial_arguments, print_configs
from src.utils.dataset_utils import load_train_dataset
from trl import GRPOConfig, GRPOTrainer
from src.utils.reward_verifiers import LLMVerifier
from src.utils.gen_utils import to_chat_prompt

from datetime import datetime
import random
import argparse


def main(config, model_config, verifier_model_config, strict_verifier_model_config, verifier_args, strict_verifier_args, config_args):
    start_time = datetime.now()
    verbose = config_args.verbose
    print(f"Script started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print_configs(config, model_config, verifier_model_config, strict_verifier_model_config)

    # Enable reasoning if specified in the model configurations
    if verifier_model_config and verifier_model_config.get("model", {}).get("reasoning"):
        verifier_args["enable_reasoning"] = True
        verifier_args["reasoning_parser"] = verifier_model_config["model"]["reasoning_parser"]
    if strict_verifier_model_config and strict_verifier_model_config.get("model", {}).get("reasoning"):
        strict_verifier_args["enable_reasoning"] = True
        strict_verifier_args["reasoning_parser"] = strict_verifier_model_config["model"]["reasoning_parser"]

    verifier_model_name = config['verifier_model']
    strict_verifier_model_name = config['strict_verifier_model']


    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        output_dir = f"train_{slurm_job_id}"
    else:
        output_dir = f"train_{random.randint(10000000, 99999999)}"

    output_dirpath = f"{config['output_base_path']}/train/{config['model']}/{config['dataset']}/{output_dir}"
    print(f"Output directory: {output_dirpath}")
    os.makedirs(output_dirpath, exist_ok=True)

    dataset = load_train_dataset(config, task_id=0, num_tasks=1)
    d = load_train_dataset(config, task_id=0, num_tasks=2)

    breakpoint()

    dataset = dataset.map(to_chat_prompt)

    

    with open(f"verifier_hosts_{config_args.verifier_server_id}.txt") as f:
        hosts_ports = [tuple(line.strip().split(":")) for line in f]

    # hosts_ports now looks like [("ip-10-0-0-5", "8000"), ("ip-10-0-0-6", "8000"), ...]
    verifier_reward = LLMVerifier(
            [(h, int(p)) for h, p in hosts_ports],
            verifier_model_name,
            strict_verifier_model_name,
            dataset,
            config)

    #TODO: Modify this for other models, eager is only for gemma3 i think
    # Load Gemma 3 model with the recommended *eager* attention implementation
    # model = Gemma3ForCausalLM.from_pretrained(
    #     model_name,
    #     attn_implementation="eager",  # avoid SDPA for Gemma 3
    #     torch_dtype="bfloat16",        # keep dtype consistent with bf16 training
    # )


    training_args = GRPOConfig(output_dir=output_dirpath, logging_steps=10)

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        args=training_args,
        reward_funcs  = verifier_reward,
        train_dataset=dataset,
    )
    trainer.train()

if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(__file__)) # Adjust if structure changes
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        default="configs/train_config.yaml",
        type=str,
        help="Config Path",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Verbose output",
    )
    config_args = parser.parse_args()
    config_args.config_path = os.path.join(repo_root, config_args.config_path)

    # Load configuration using the parsed path
    config, model_config, model_path, verifier_model_config, verifier_model_path, strict_verifier_model_config, strict_verifier_model_path = load_and_validate_config(config_args, repo_root)

    print("Parsing verifier arguments...")
    verifier_args = parse_vllm_args(verifier_model_path)
    print("Verifier arguments parsed.")

    print("Parsing strict verifier arguments...")
    strict_verifier_args = parse_vllm_args(strict_verifier_model_path)
    print("Strict Verifier arguments parsed.")

    if config_args.verbose:
        print(f"Verifier Args: {verifier_args}")
        print(f"Strict Verifier Args: {strict_verifier_args}")

    main(config, model_config, verifier_model_config, strict_verifier_model_config, verifier_args, strict_verifier_args, config_args)