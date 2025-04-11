import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset_utils import load_eval_dataset
from utils.gen_utils import generate_solutions
from utils.eval_utils import evaluate_model
from utils.loading_utils import load_models
import argparse
import random
import numpy as np
import json
import os
from tqdm import trange
import yaml

from datetime import datetime
import torch.distributed as dist

def main():
    """
    Main function to run the complete training and evaluation pipeline.

    The process consists of:
      1. Loading the pre-trained model and tokenizer.
      2. Evaluating the initial model performance (before any finetuning).
      3. Performing reinforcement learning (GRPO) finetuning with MAV as the reward function, evaluating the model after each epoch.
      4. Saving the finetuned model and tokenizer.

    """
    start_str_for_file = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Parser arguments that loads a config file but also allows for command line arguments to override the common config values
    parser = argparse.ArgumentParser(description="Self-Improvement via GRPO and MAV")
    parser.add_argument("--task-id", type=int, required=True, help="Unique task ID (e.g., SLURM_PROCID, 0-based).")
    parser.add_argument("--num-tasks", type=int, required=True, help="Total number of client tasks.")
    parser.add_argument("--config-file", type=str, required=True, help="Path to the main configuration YAML file.")
    parser.add_argument("--shared-dir", type=str, required=True, help="Path to the shared directory for coordination files (e.g., host files).")
    args_cli = parser.parse_args()

    # Load config file
    with open(args_cli.config_file, 'r') as f:
        config = yaml.safe_load(f)

    config['task_id'] = args_cli.task_id
    config['num_tasks'] = args_cli.num_tasks
    config['shared_dir'] = args_cli.shared_dir

    random.seed(config['seed'])
    np.random.seed(config['seed'])


    # Check if eval_mode is a list and handle accordingly
    if isinstance(config['eval_mode'], list):
        eval_modes_str = "_".join(config['eval_mode'])
    else:
        eval_modes_str = config['eval_mode']

    # Check if output_dir is provided or generate one dynamically
    if config['output_dir'] is None:
        output_dir = f"{start_str_for_file}_{config['model']}_{config['dataset']}_{eval_modes_str}"
    else:
        output_dir = config['output_dir']
    
    # Use the base path from config
    output_dirpath = f"{config['output_base_path']}/{output_dir}"
    os.makedirs(output_dirpath, exist_ok=True)
    print(f"Output directory: {output_dirpath}")

    print("Logging config to YAML file...")
    with open(f"{output_dirpath}/config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # # Load model and config
    # # TODO: Load model and config
    generator_model, verifier_model = load_models(config)
    # Load the eval dataset 
    eval_dataset = load_eval_dataset(config)

    evaluate_model(config, generator_model, generator_tokenizer, verifier_model, verifier_tokenizer, eval_dataset, output_dirpath)


if __name__ == "__main__":
    main()