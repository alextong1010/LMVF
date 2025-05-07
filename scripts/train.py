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
from src.utils.gen_utils import make_to_chat_prompt

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

    dataset = dataset.map(make_to_chat_prompt(config))

    def reward_num_unique_chars(completions, **kwargs):
        # print(completions)
        return [len(set(c)) for c in completions]

        # Need to modify vllm_server_host, vllm_server_port, vllm_server_timeout

    breakpoint()

    training_args = GRPOConfig(
        output_dir=output_dirpath, 
        per_device_train_batch_size=config['per_device_train_batch_size'],
        max_steps=config['max_steps'], 
        logging_steps=config['logging_steps'],
        save_steps=config['save_steps'],
        save_total_limit=config['save_total_limit'],
        seed=config['seed'],
        max_prompt_length=config['max_prompt_length'],
        num_generations=config['num_generations'],
        max_completion_length=model_config['model']['max_new_tokens'],
        temperature=model_config['model']['temperature'],
        top_k=model_config['model']['top_k'],
        top_p=model_config['model']['top_p'],
        min_p=model_config['model']['min_p'],
        use_vllm=config_args.use_vllm,
        vllm_server_host="holygpu8a22202",
        vllm_server_port=8000,
        vllm_server_timeout=600,
        # vllm_gpu_memory_utilization=model_config['model'].get('vllm_gpu_memory_utilization', 0.3),
        # vllm_tensor_parallel_size=model_config['model'].get('tensor_parallel_size', 1)
    )
    breakpoint()
    trainer = GRPOTrainer(
        model=model_config['model']['path'],
        args=training_args,
        reward_funcs = reward_num_unique_chars,
        train_dataset=dataset,
    )
    breakpoint()

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
        "--use-vllm",
        default=False,
        action="store_true",
        help="Use VLLM",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--local",
        default=False,
        action="store_true",
        help="If true, not using slurm",
    )
    # Use parse_known_args() to separate script args from others (like vLLM args)
    config_args, remaining_args = parser.parse_known_args()
    config_args.config_path = os.path.join(repo_root, config_args.config_path)

    if config_args.local:
        print("Local mode, not using slurm")
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "holygpu8a22204" #"localhost"
        os.environ["MASTER_PORT"] = "8000"
    else:
        print("Using slurm")

    # Load configuration using the parsed path
    config, model_config, model_path, verifier_model_config, verifier_model_path, strict_verifier_model_config, strict_verifier_model_path = load_and_validate_config(config_args, repo_root)

    print("Parsing verifier arguments...")
    # Pass remaining_args to parse_vllm_args
    verifier_args = parse_vllm_args(verifier_model_path, remaining_args=remaining_args)
    print("Verifier arguments parsed.")

    print("Parsing strict verifier arguments...")
    # Pass remaining_args to parse_vllm_args
    strict_verifier_args = parse_vllm_args(strict_verifier_model_path, remaining_args=remaining_args)
    print("Strict Verifier arguments parsed.")

    if config_args.verbose:
        print(f"Verifier Args: {verifier_args}")
        print(f"Strict Verifier Args: {strict_verifier_args}")

    main(config, model_config, verifier_model_config, strict_verifier_model_config, verifier_args, strict_verifier_args, config_args)