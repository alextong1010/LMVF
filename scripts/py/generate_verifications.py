import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import yaml
from src.utils.verifier_runner import VerifierRunner
from src.utils.dataset_manager import DatasetManager
import torch.distributed as dist

from datetime import datetime
from termcolor import colored


def main(repo_root, args):
    # Log start time
    start_time = datetime.now()
    print(colored(f"Verification of solutions started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}, Task {args.task_id}, Num Tasks {args.num_tasks}", "white", attrs=["bold"]))

    # Load generation config from args
    config_path = os.path.join(repo_root, args.config_path)
    print(colored(f"Loading config from {config_path}", "yellow", attrs=["bold"]))
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
            
    output_dirpath = f"{config['base_config']['output_base_path']}/{config['base_config']['dataset']}/gen/{config['base_config']['gen_seed']}"
    print(colored(f"Output directory: {output_dirpath}", "yellow", attrs=["bold"]))
    os.makedirs(output_dirpath, exist_ok=True)

    # Initialize dataset loader
    dataset_manager = DatasetManager(config, args.task_id, args.num_tasks)
    dataset_manager.load_dataset("test", filepath=f"{output_dirpath}/{config['base_config']['solutions_file_name'].format(task_id=args.task_id)}")
    
    # Initialize verifier
    verifier = VerifierRunner(config, dataset_manager, output_dirpath)
    print(colored(f"Using model(s) for verification: {verifier.ver_models}", "yellow"))
    
    try:
        verifier.generate_init_verifications()
        verifier.generate_strict_verifications()
        verifier.save_verifications()
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
        default="configs/verify_config.yaml",
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