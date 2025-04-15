# SPDX-License-Identifier: Apache-2.0
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vllm import LLM
from src.utils.loading_utils import load_and_validate_config, parse_vllm_args, parse_initial_arguments
from src.utils.dataset_utils import load_eval_dataset
from src.utils.gen_utils import generate_solutions
from src.utils.eval_utils import evaluate_problem
from tqdm import trange
from datetime import datetime
from collections import defaultdict
import json
import copy

def main(config: dict, model_config: dict, args: dict, verifier_args: dict, task_id: int, num_tasks: int, verbose: bool):
    start_time = datetime.now()
    print(f"Script started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}, Task {task_id}, Num Tasks {num_tasks}")

    if model_config.get("reasoning"):
        args["enable_reasoning"] = True
        args["reasoning_parser"] = model_config["reasoning_parser"]
    
    start_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Check if eval_mode is a list and handle accordingly
    if isinstance(config['eval_mode'], list):
        eval_modes_str = "_".join(config['eval_mode'])
    else:
        eval_modes_str = config['eval_mode']

    if "bon-mav" in config['eval_mode']:
        verifier_str = config.get("verifier_model", "self")
        eval_modes_str += f"_verifier_{verifier_str}"

    output_dir = f"{eval_modes_str}_{start_str}"

    
    # Use the base path from config
    output_dirpath = f"{config['output_base_path']}/{config['model']}/{config['dataset']}/{output_dir}"
    os.makedirs(output_dirpath, exist_ok=True)

    # Create the generator LLM
    llm = LLM(**args)
    # Create sampling params object
    sampling_params = llm.get_default_sampling_params()
    # override given config file
    sampling_params.max_tokens = model_config.get('model', {}).get('max_new_tokens')
    sampling_params.n = config['num_generations']

    # Load the dataset
    dataset = load_eval_dataset(config, task_id, num_tasks)
    total_eval = len(dataset)
    # Break into batches and generate solutions
    for i in trange(0, total_eval, config['batch_size'], desc=f"Eval"):
        batch = dataset[i:i + config['batch_size']]
        data_with_generated_solutions = generate_solutions(config, batch, llm, sampling_params)

        # Save data_with_generated_solutions
        with open(f"{output_dirpath}/task_{task_id}_eval_{i}-{i+config['batch_size']}.json", "w") as f:
            json.dump(data_with_generated_solutions, f)
    # Load the verifier if needed, else use the generator
    if not "bon-mav" in config['eval_mode']:
        verifier_llm = None
        verifier_sampling_params = None
        strict_verifier_sampling_params = None
    elif verifier_args:
        # Delete the generator LLM object to free memory before loading the verifier
        del llm
        verifier_llm = LLM(**verifier_args)
        verifier_sampling_params = verifier_llm.get_default_sampling_params()
        verifier_sampling_params.max_tokens = verifier_model_config.get('model', {}).get('max_new_tokens')
        strict_verifier_sampling_params = copy.deepcopy(verifier_sampling_params)
        strict_verifier_sampling_params.max_tokens = 16
    else:
        verifier_llm = llm
        verifier_sampling_params = sampling_params
        verifier_sampling_params.n = 1
        strict_verifier_sampling_params = copy.deepcopy(verifier_sampling_params)
        strict_verifier_sampling_params.max_tokens = 16

    # Evaluate the generated solutions
    # Get all of the json files from output_dirpath that start with task_<task_id> and end with .json
    json_files = [f for f in os.listdir(output_dirpath) if f.startswith(f"task_{task_id}") and f.endswith(".json")]
    len_json_files = len(json_files)
    print(f"Evaluating {len_json_files} batches...")

    correct_counts = defaultdict(int)
    for i in trange(0, len_json_files, desc=f"Verifying"):
        file = json_files[i]
        print(f"Evaluating {file}...")
        with open(os.path.join(output_dirpath, file), 'r') as f:
            data_with_generated_solutions = json.load(f)
        results, data_with_generated_solutions = evaluate_problem(config, data_with_generated_solutions, verifier_llm, verifier_sampling_params, strict_verifier_sampling_params)
        # Save updated data back to the original file
        with open(os.path.join(output_dirpath, file), 'w') as f:
            json.dump(data_with_generated_solutions, f)
        for eval_mode, count in results.items():
            correct_counts[eval_mode] += count

    # Save count, total_eval, and accuracy to a dictionary before printing and saving to a file
    accuracy_dict = {}
    for eval_mode, count in correct_counts.items():
        accuracy = count / total_eval
        accuracy_dict[eval_mode] = {
            "correct_count": count,
            "total_eval": total_eval,
            "accuracy": accuracy
        }
        print(f"[Task {task_id}] {eval_mode} Accuracy: {accuracy:.2%} ({count} / {total_eval})")


    with open(os.path.join(output_dirpath, f"accuracy_task_{task_id}.json"), "w") as f:
        json.dump(accuracy_dict, f, indent=2)

    # Create the cfg directory if it doesn't exist
    cfg_dir = os.path.join(output_dirpath, "cfg")
    os.makedirs(cfg_dir, exist_ok=True)

    # Save config, model_config, args, verifier_args, task_id, num_tasks
    with open(os.path.join(cfg_dir, "config.json"), "w") as f:
        json.dump(config, f)
    with open(os.path.join(cfg_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f)
    with open(os.path.join(cfg_dir, "args.json"), "w") as f:
        json.dump(args, f)
    with open(os.path.join(cfg_dir, "verifier_args.json"), "w") as f:
        json.dump(verifier_args, f)
    with open(os.path.join(cfg_dir, "task_id.json"), "w") as f:
        json.dump(task_id, f)
    with open(os.path.join(cfg_dir, "num_tasks.json"), "w") as f:
        json.dump(num_tasks, f)
        

    end_time = datetime.now()
    print(f"Script ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {end_time - start_time}")

if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(__file__)) # Adjust if structure changes
    config_args, remaining_vllm_args = parse_initial_arguments(repo_root)

    task_id = config_args.task_id
    num_tasks = config_args.num_tasks
    verbose = config_args.verbose

    # Load configuration using the parsed path
    config, model_config, model_path, verifier_model_config, verifier_model_path = load_and_validate_config(config_args, repo_root)
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
    main(config, model_config, args, verifier_args, task_id, num_tasks, verbose)