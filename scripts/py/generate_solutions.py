import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import yaml
from src.utils.generation_runner import GenerationRunner
from src.utils.dataset_manager import DatasetManager
import torch.distributed as dist

from datetime import datetime
from termcolor import colored


def main(repo_root, args):
    # Log start time
    start_time = datetime.now()
    print(colored(f"Solution Generation started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}, Task {args.task_id}, Num Tasks {args.num_tasks}", "white", attrs=["bold"]))

    # Load generation config from args
    config_path = os.path.join(repo_root, args.config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
            
    output_dirpath = f"{config['base_config']['output_base_path']}/{config['base_config']['dataset']}"
    print(colored(f"Output directory: {output_dirpath}", "yellow", attrs=["bold"]))
    os.makedirs(output_dirpath, exist_ok=True)

    # Initialize dataset loader
    dataset_manager = DatasetManager(config, args.task_id, args.num_tasks)
    dataset_manager.load_dataset("test")
    dataset_manager.extract_ground_truth()
    
    # Initialize generator
    generator = GenerationRunner(config)
    print(colored(f"Using model(s) for generation: {generator.gen_models}", "yellow"))
    
    try:
        generator.generate_solutions(dataset_manager)
        dataset_manager.save_dataset()
    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
                print(colored("NCCL process group destroyed successfully", "green"))
            except Exception as e:
                print(colored(f"Warning: Failed to destroy process group: {e}", "yellow"))
    
    # dataset_manager.save_dataset(f"{output_dirpath}/gen/{output_file_name}")

if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # Adjust if structure changes
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        default="configs/gen_config.yaml",
        type=str,
        help="Config Path",
    )
    parser.add_argument(
        "--task-id",
        default=0,
        type=int,
        help="Task ID (defaults to 0 for single gpu tasks)",
    )
    parser.add_argument(
        "--num-tasks",
        default=1,
        type=int,
        help="Number of tasks",
    )
    args = parser.parse_args()
    main(repo_root, args)