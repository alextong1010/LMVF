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

def main(repo_root, args):
    # Log start time
    start_time = datetime.now()
    print(colored(f"Evaluation started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}, Task {args.task_id}, Num Tasks {args.num_tasks}", "white", attrs=["bold"]))
    
    # Load generation config from args
    config_path = os.path.join(repo_root, args.config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    output_dirpath = f"{config['base_config']['solutions_dir']}/{config['base_config']['dataset']}"
    print(colored(f"Output directory: {output_dirpath}", "yellow", attrs=["bold"]))
    os.makedirs(output_dirpath, exist_ok=True)

    dataset_manager = DatasetManager(config, args.task_id, args.num_tasks)
    dataset_manager.load_dataset("test", eval=True)

    assert_dataset_contains_req_models(dataset_manager, config['base_config']['models'])
    
    # Create evaluator
    evaluator = EvaluationRunner(config)

    # Run evaluator
    evaluator.run_eval_scenarios(dataset_manager)

    # Save dataset
    breakpoint()
    # dataset_manager.save_dataset(output_dirpath)

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