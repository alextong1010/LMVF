import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import yaml
from src.utils.dataset_manager import DatasetManager
from src.utils.util import assert_dataset_contains_req_models
from src.utils.evaluation_runner import EvaluationRunner

from datetime import datetime
from termcolor import colored
import random

def main(repo_root, args):
    # Log start time
    start_time = datetime.now()
    print(colored(f"Evaluation started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}, Task {args.task_id}, Num Tasks {args.num_tasks}", "white", attrs=["bold"]))
    
    # Create a random 8 digit seed
    run_seed = random.randint(0, 100000000)
    print(colored(f"Random run seed: {run_seed}", "white", attrs=["bold"]))

    # Load generation config from args
    config_path = os.path.join(repo_root, args.config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    output_dirpath = f"{config['base_config']['output_base_path']}/{config['base_config']['dataset']}/eval/{run_seed}"
    print(colored(f"Output directory: {output_dirpath}", "yellow", attrs=["bold"]))
    os.makedirs(output_dirpath, exist_ok=True)

    solutions_file_path = f"{config['base_config']['output_base_path']}/{config['base_config']['dataset']}/gen/{config['base_config']['gen_seed']}/{config['base_config']['solutions_file_name'].format(task_id=args.task_id)}"
    if not os.path.exists(solutions_file_path):
        raise FileNotFoundError(f"Solutions file {solutions_file_path}")

    dataset_manager = DatasetManager(config, args.task_id, args.num_tasks)
    dataset_manager.load_dataset("test", filepath=solutions_file_path)

    assert_dataset_contains_req_models(dataset_manager, config['base_config']['models'])
    
    # Create evaluator
    evaluator = EvaluationRunner(config, dataset_manager, output_dirpath)

    # Run evaluator
    evaluator.run_eval_scenarios()
    breakpoint()

    # Save results
    evaluator.save_results()

    breakpoint()

if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # Adjust if structure changes
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        default="configs/eval_config.yaml",
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