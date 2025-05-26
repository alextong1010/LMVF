# SPDX-License-Identifier: Apache-2.0
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vllm import LLM
from src.utils.loading_utils import load_and_validate_config, parse_vllm_args, parse_initial_arguments
from src.utils.dataset_utils import load_eval_dataset
from src.utils.gen_utils import generate_solutions
from src.utils.eval_utils import run_initial_verification_batch, run_strict_verification_and_evaluate_batch, evaluate_problem_pass_at_n, run_borda_evaluation_batch
from tqdm import trange
from datetime import datetime
from collections import defaultdict
import json
import random
import torch.distributed as dist

import gc
import torch
from vllm.distributed.parallel_state import destroy_model_parallel

# ----------------------------------------------------------------------
# Utility that reliably frees a vLLM instance and its KV cache
# ----------------------------------------------------------------------
def cleanup_llm(llm):
    """
    Tear down all model‑parallel buffers, free CUDA memory, and close
    NCCL/Ray resources that keep the KV cache alive.
    """
    if llm is None:
        return None

    # 1. Let vLLM dismantle model‑parallel state
    destroy_model_parallel()

    # 2. Drop the last Python reference
    del llm

    # 3. Run the CPython GC & clear PyTorch’s allocator
    gc.collect()
    torch.cuda.empty_cache()

    # 4. Close the NCCL pg if it exists
    if dist.is_initialized():
        dist.destroy_process_group()

    return None

def main(config: dict, model_config: dict, verifier_model_config: dict, strict_verifier_model_config: dict, args: dict, args_2: dict, verifier_args: dict, strict_verifier_args: dict, task_id: int, num_tasks: int, verbose: bool):
    start_time = datetime.now()
    print(f"Script started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}, Task {task_id}, Num Tasks {num_tasks}")
    print(f"Using model: {config['model']}, verifier model: {config.get('verifier_model')}, strict verifier model: {config.get('strict_verifier_model')}")
    
    print("-"*100)
    print(f"Model config: {model_config}")
    print("-"*100)
    print(f"Verifier model config: {verifier_model_config}")
    print("-"*100)
    print(f"Strict verifier model config: {strict_verifier_model_config}")
    print("-"*100)
        
    # Enable reasoning if specified in the model configurations
    if model_config.get("model", {}).get("reasoning"):
        args["enable_reasoning"] = True
        args["reasoning_parser"] = model_config["model"]["reasoning_parser"]
    if verifier_model_config and verifier_model_config.get("model", {}).get("reasoning"):
        verifier_args["enable_reasoning"] = True
        verifier_args["reasoning_parser"] = verifier_model_config["model"]["reasoning_parser"]
    if strict_verifier_model_config and strict_verifier_model_config.get("model", {}).get("reasoning"):
        strict_verifier_args["enable_reasoning"] = True
        strict_verifier_args["reasoning_parser"] = strict_verifier_model_config["model"]["reasoning_parser"]

    # Determine evaluation modes
    if isinstance(config['eval_mode'], list):
        eval_modes = config['eval_mode']
        eval_modes_str = "_".join(eval_modes)
    else:
        eval_modes = [config['eval_mode']]
        eval_modes_str = config['eval_mode']

    needs_bon_mav = "bon-mav" in eval_modes
    needs_pass_at_n = any(mode.startswith("pass@") for mode in eval_modes)
    needs_borda = "borda" in eval_modes

    if needs_bon_mav:
        verifier_str = config.get("verifier_model", "None")
        strict_verifier_str = config.get("strict_verifier_model", "None")
        eval_modes_str += f"_verifier_{verifier_str}_strict_{strict_verifier_str}"

    if needs_borda:
        verifier_str = config.get("verifier_model", "None")
        eval_modes_str += f"_verifier_{verifier_str}"

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        output_dir = f"{eval_modes_str}_{slurm_job_id}"
    else:
        output_dir = f"{eval_modes_str}_{random.randint(10000000, 99999999)}"

    output_dirpath = f"{config['output_base_path']}/{config['model']}/{config['dataset']}/{output_dir}"
    print(f"Output directory: {output_dirpath}")
    os.makedirs(output_dirpath, exist_ok=True)

    # for gen_num in range(1, config['num_generators'] + 1):
    # --- Stage 1: Generate Solutions ---
    print("Stage 1: Generating Solutions...")    
    llm = LLM(**args)
    sampling_params = llm.get_default_sampling_params()
    sampling_params.max_tokens = model_config.get('model', {}).get('max_new_tokens')
    sampling_params.n = config['num_generations']
    dataset = list(load_eval_dataset(config, task_id, num_tasks))
    total_eval = len(dataset)
    
    print(f"Generating solutions for {total_eval} problems...")
    for i in trange(0, total_eval, config['batch_size'], desc=f"Generating Task {task_id}"):
        batch = dataset[i:i + config['batch_size']]
        data_with_generated_solutions = generate_solutions(config, model_config, batch, llm, sampling_params)
        with open(f"{output_dirpath}/gen_1_task_{task_id}_eval_{i}-{i+config['batch_size']}.json", "w") as f:
            json.dump(data_with_generated_solutions, f)
    
    llm = cleanup_llm(llm)

    print("Part 2 Generating Solutions")
    llm = LLM(**args_2)
    sampling_params = llm.get_default_sampling_params()
    sampling_params.max_tokens = model_config.get('model', {}).get('max_new_tokens')
    sampling_params.n = config['num_generations']
    dataset = list(load_eval_dataset(config, task_id, num_tasks))
    total_eval = len(dataset)
    
    print(f"Generating solutions for {total_eval} problems...")
    for i in trange(0, total_eval, config['batch_size'], desc=f"Generating Task {task_id}"):
        batch = dataset[i:i + config['batch_size']]
        data_with_generated_solutions = generate_solutions(config, model_config, batch, llm, sampling_params, gen_num=2)
        with open(f"{output_dirpath}/gen_2_task_{task_id}_eval_{i}-{i+config['batch_size']}.json", "w") as f:
            json.dump(data_with_generated_solutions, f)

    breakpoint()
    
    # Check if verifier_llm is the same as llm
    verifier_llm = None
    if needs_bon_mav:
        if verifier_args == args:
            verifier_llm = llm
            print("Using the same LLM for both generation and initial verification.")
        else:
            llm = cleanup_llm(llm)
            print("Loading Verifier LLM...")
            verifier_llm = LLM(**verifier_args)
        verifier_sampling_params = verifier_llm.get_default_sampling_params()
        verifier_sampling_params.max_tokens = verifier_model_config.get('model', {}).get('max_new_tokens')
        verifier_sampling_params.n = 1

        print("\nStage 2: Initial Verification...")
        json_files = [f for f in os.listdir(output_dirpath) if f.startswith(f"gen_1_task_{task_id}") and f.endswith(".json")]
        for file in trange(len(json_files), desc=f"Initial Verification Task {task_id}"):
            filepath = os.path.join(output_dirpath, json_files[file])
            with open(filepath, 'r') as f:
                batch_data = json.load(f)
            
            if batch_data and 'verifier_responses' in batch_data[0]:
                print(f"Skipping initial verification for {json_files[file]}, already found 'verifier_responses'.")
                continue

            updated_batch_data = run_initial_verification_batch(
                config, verifier_model_config, batch_data, verifier_llm, verifier_sampling_params
            )
            with open(filepath, 'w') as f:
                json.dump(updated_batch_data, f)
        
        # Check if strict_verifier_llm is the same as verifier_llm
        strict_verifier_llm = None
        if strict_verifier_args == verifier_args:
            strict_verifier_llm = verifier_llm
            print("Using the same LLM for both initial and strict verification.")
        else:
            verifier_llm = cleanup_llm(verifier_llm)
            print("Loading Strict Verifier LLM...")
            strict_verifier_llm = LLM(**strict_verifier_args)
        strict_verifier_sampling_params = strict_verifier_llm.get_default_sampling_params()
        if strict_verifier_model_config.get("model", {}).get("reasoning"):
            strict_verifier_sampling_params.max_tokens = 2048
            strict_verifier_model_config["model"]["max_new_tokens"] = 2048

        else:
            strict_verifier_sampling_params.max_tokens = 16
            strict_verifier_model_config["model"]["max_new_tokens"] = 16

        print("\nStage 3: Strict Verification...")
        correct_counts = defaultdict(int)
        for file in trange(len(json_files), desc=f"Strict Verification & Eval Task {task_id}"):
            filepath = os.path.join(output_dirpath, json_files[file])
            with open(filepath, 'r') as f:
                batch_data = json.load(f)

            if batch_data and 'verifier_approval_bools' in batch_data[0]:
                print(f"Skipping strict verification for {json_files[file]}, already found 'verifier_approval_bools'.")
                results = run_strict_verification_and_evaluate_batch(
                    config, strict_verifier_model_config, batch_data, None, None, skip_llm_call=True
                )
            else:
                results = run_strict_verification_and_evaluate_batch(
                    config, strict_verifier_model_config, batch_data, strict_verifier_llm, strict_verifier_sampling_params
                )

            with open(filepath, 'w') as f:
                json.dump(results['updated_batch_data'], f)

            correct_counts["bon-mav"] += results['correct_count']

        strict_verifier_llm = cleanup_llm(strict_verifier_llm)
        print("Stage 3 Complete.")

    if needs_pass_at_n:
        print("\nStage 4: Pass@N Evaluation...")
        json_files = [f for f in os.listdir(output_dirpath) if f.startswith(f"task_{task_id}") and f.endswith(".json")]
        pass_n_modes = [mode for mode in eval_modes if mode.startswith("pass@")]

        for file in trange(len(json_files), desc=f"Pass@N Eval Task {task_id}"):
            filepath = os.path.join(output_dirpath, json_files[file])
            with open(filepath, 'r') as f:
                batch_data = json.load(f)

            for mode in pass_n_modes:
                try:
                    n = int(mode.split("@")[1])
                    correct_count = evaluate_problem_pass_at_n(config, batch_data, n)
                    correct_counts[mode] += correct_count
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid pass@n format: {mode}")
        print("Stage 4 Complete.")


    print("\nFinal Results:")
    accuracy_dict = {}
    for eval_mode in eval_modes:
        count = correct_counts[eval_mode]
        if total_eval > 0:
            accuracy = count / total_eval
            accuracy_dict[eval_mode] = {
                "correct_count": count,
                "total_eval": total_eval,
                "accuracy": accuracy
            }
            print(f"[Task {task_id}] {eval_mode} Accuracy: {accuracy:.2%} ({count} / {total_eval})")
        else:
            accuracy_dict[eval_mode] = {
                "correct_count": 0,
                "total_eval": 0,
                "accuracy": 0
            }
            print(f"[Task {task_id}] {eval_mode} Accuracy: N/A (0 / 0)")

    with open(os.path.join(output_dirpath, f"accuracy_task_{task_id}.json"), "w") as f:
        json.dump(accuracy_dict, f, indent=2)

    cfg_dir = os.path.join(output_dirpath, "cfg")
    os.makedirs(cfg_dir, exist_ok=True)

    with open(os.path.join(cfg_dir, "config.json"), "w") as f:
        json.dump(config, f)
    with open(os.path.join(cfg_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f)
    with open(os.path.join(cfg_dir, "verifier_model_config.json"), "w") as f:
        json.dump(verifier_model_config, f)
    with open(os.path.join(cfg_dir, "strict_verifier_model_config.json"), "w") as f:
        json.dump(strict_verifier_model_config, f)
    with open(os.path.join(cfg_dir, "args.json"), "w") as f:
        json.dump(args, f)
    with open(os.path.join(cfg_dir, "verifier_args.json"), "w") as f:
        json.dump(verifier_args, f)
    with open(os.path.join(cfg_dir, "strict_verifier_args.json"), "w") as f:
        json.dump(strict_verifier_args, f)
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

    config, model_config, model_path, verifier_model_config, verifier_model_path, strict_verifier_model_config, strict_verifier_model_path = load_and_validate_config(config_args, repo_root)

    print("Parsing generator arguments...")
    args = parse_vllm_args(model_path, remaining_vllm_args) # Pass remaining args
    print("Generator arguments parsed.")

    args_2 = parse_vllm_args("microsoft/Phi-3-mini-4k-instruct", remaining_vllm_args)

    verifier_args = None
    if verifier_model_path:
        print("Parsing verifier arguments...")
        verifier_args = parse_vllm_args(verifier_model_path, remaining_vllm_args)
        print("Verifier arguments parsed.")
    elif "bon-mav" in (config['eval_mode'] if isinstance(config['eval_mode'], list) else [config['eval_mode']]):
        print("Warning: 'bon-mav' evaluation specified but no verifier model config provided.")

    strict_verifier_args = None
    if strict_verifier_model_path:
        print("Parsing strict verifier arguments...")
        strict_verifier_args = parse_vllm_args(strict_verifier_model_path, remaining_vllm_args)
        print("Strict Verifier arguments parsed.")
    elif "bon-mav" in (config['eval_mode'] if isinstance(config['eval_mode'], list) else [config['eval_mode']]):
        print("Warning: 'bon-mav' evaluation specified but no strict verifier model config provided.")

    if verbose:
        print(f"Generator Args: {args}")
        print(f"Verifier Args: {verifier_args}")
        print(f"Strict Verifier Args: {strict_verifier_args}")
    main(config, model_config, verifier_model_config, strict_verifier_model_config, args, args_2, verifier_args, strict_verifier_args, task_id, num_tasks, verbose)


    