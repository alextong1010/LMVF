import json
import argparse
import time
from datetime import datetime
import random
import numpy as np
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.dataset_utils import load_eval_dataset
from utils.model_registry import get_full_model_name
import torch
from utils.gen_utils import generate_solutions
from utils.dataset_utils import check_correct_answer
from utils.vera_utils import generate_verifier_approvals

from collections import defaultdict
from tqdm import trange


def evaluate_problem(config, data_with_generated_solutions, verifier_model=None, verifier_tokenizer=None, accelerator=None):
    """
    Evaluates the model on a single problem or batch of problems using the pass@n evaluation metric.

    Args:
        config (dict): The configuration.
        data_with_generated_solutions (list): List of dictionaries containing "generated_answers" and "gt_answer".
        verifier_model: The language model to use for verification.
        verifier_tokenizer: The tokenizer for the verifier model.
        accelerator: Accelerator for distributed processing.

    Returns:
        dict: A dictionary where keys are eval modes and values are the number of correct predictions.
    """
    # Handle the case where eval_mode is a list
    eval_modes = config['eval_mode'] if isinstance(config['eval_mode'], list) else [config['eval_mode']]
    
    results = {}
    for eval_mode in eval_modes:
        if eval_mode.startswith("pass@"):
            try:
                n = int(eval_mode.split("@")[1])
                correct_count = evaluate_problem_pass_at_n(config, data_with_generated_solutions, n)
            except (ValueError, IndexError):
                raise ValueError(f"Invalid eval_mode(s) format: {eval_mode}. Expected format: pass@n where n is a number.")
        elif eval_mode == "bon-mav": # BoN-MAV runs through all of the generated answers
            correct_count = evaluate_problem_bon_mav(config, data_with_generated_solutions, verifier_model, verifier_tokenizer, accelerator)
        else:
            raise ValueError(f"Invalid eval_mode(s): {eval_mode}")
        
        results[eval_mode] = correct_count
    
    return results

def evaluate_problem_bon_mav(config, data_with_generated_solutions, verifier_model=None, verifier_tokenizer=None, accelerator=None):
    """
    Evaluates the model on a single problem or batch of problems using the bon-mav evaluation metric.
    
    Args:
        config (dict): The configuration.
        data_with_generated_solutions (list): List of dictionaries containing generated solutions.
        verifier_model: The language model to use for verification.
        verifier_tokenizer: The tokenizer for the verifier model.
        accelerator: Accelerator for distributed processing.
    """
    correct_count = 0 

    for d in data_with_generated_solutions:
        gt_answer, generated_answers, verifier_approval_bools, data = generate_verifier_approvals(
            config, d, verifier_model, verifier_tokenizer, accelerator
        )
        approval_sums = [sum(sublist) for sublist in verifier_approval_bools]

        # Get the index of the sublist with the highest sum
        best_index = approval_sums.index(max(approval_sums))

        # Get the best generated answer based on the highest approval score
        best_generated_answer = generated_answers[best_index]

        if config['verbose']:
            print(f"gt_answer: {gt_answer}")
            print(f"generated_answers: {generated_answers}")
            print(f"verifier_approval_bools: {verifier_approval_bools}")
            print(f"best_index: {best_index}")
            print(f"best_generated_answer: {best_generated_answer}")
        # Compare it to the ground truth answer
        is_correct = check_correct_answer(best_generated_answer, gt_answer, config['dataset'])
        if is_correct:
            correct_count += 1
    
    return correct_count

def evaluate_problem_pass_at_n(config, data_with_generated_solutions, n):
    """
    Evaluates the model on a single problem or batch of problems using the pass@n evaluation metric.

    Args:
        args (argparse.Namespace): The arguments.
        data_with_generated_solutions (list): List of dictionaries containing "generated_answers" and "gt_answer".
        n (int): The number of generated answers to consider for evaluation (e.g., 1 for pass@1, 8 for pass@8).

    Returns:
        int: The number of correct predictions.
    """
    assert config['num_generations'] >= n, f"num_generations must be {n} or greater for pass@{n} evaluation"
    if config['num_generations'] > n:
        print(f"Warning: You've set num_generations to {config['num_generations']}, but eval_mode is set to pass@{n}. Only the first {n} generated answers will be considered.")

    correct_count = 0
    for d in data_with_generated_solutions:
        is_correct = False
        for i in range(n):
            is_correct = check_correct_answer(d["generated_answers"][i], d["gt_answer"], config['dataset'])
            if is_correct:
                correct_count += 1
                break
        if config['verbose']:
            print(f"Checking if any of the first {n} generated answers {d['generated_answers'][:n]} is the same as the ground truth answer {d['gt_answer']} | Result: {'Correct' if is_correct else 'Incorrect'}")

    return correct_count


def evaluate_model(config, generator_model, generator_tokenizer, verifier_model, verifier_tokenizer, eval_dataset, output_dirpath):
    """
    Evaluates the model based on the evaluation mode specified in the config.
    
    Args:
        config (dict): Configuration dictionary.
        generator_model: The language model to use for generation.
        generator_tokenizer: The tokenizer for the generator model.
        verifier_model: The language model to use for verification.
        verifier_tokenizer: The tokenizer for the verifier model.
        eval_dataset (list): List of evaluation examples (already split into chunks based on task_id).
        accelerator: Accelerator for distributed processing.
        output_dirpath (str): Directory to store evaluation results.
    """
    
    generator_model.eval()
    verifier_model.eval()
    eval_examples = list(eval_dataset)
    total_eval = len(eval_examples)
    
    print(f"Evaluating {config['dataset']} (total {total_eval} examples) using {config['eval_mode']}")

    correct_counts = defaultdict(int)

    # Break into batches and evaluate
    for i in trange(0, len(eval_examples), config['batch_size'], desc=f"Rank {rank} (eval)"):
        batch = eval_examples[i:i + config['batch_size']]
        data_with_generated_solutions = generate_solutions(config, batch, generator_model, generator_tokenizer, accelerator)

        # Get and accumulate results for multiple eval modes dynamically
        results = evaluate_problem(config, data_with_generated_solutions, verifier_model, verifier_tokenizer, accelerator)
        for eval_mode, count in results.items():
            correct_counts[eval_mode] += count

        data_idx = start_idx + i # global position for naming
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"[{now}] Rank {rank} completed eval {data_idx}-{data_idx+config['batch_size']}")
        with open(f"{output_dirpath}/eval_{data_idx}-{data_idx+config['batch_size']}_rank{rank}.json", "w") as f:
            json.dump(data_with_generated_solutions, f)

    # Convert local correct counts to tensors dynamically
    correct_tensors = {eval_mode: torch.tensor([count], dtype=torch.long, device=accelerator.device) for eval_mode, count in correct_counts.items()}
    # Reduce (sum) across all processes dynamically
    total_correct_counts = {eval_mode: accelerator.reduce(tensor, reduction="sum").item() for eval_mode, tensor in correct_tensors.items()}

    # Only print from the main process
    if accelerator.is_main_process:
        print(f"Evaluation complete on {config['dataset']} dataset using multiple eval modes:")
        for eval_mode, total_correct in total_correct_counts.items():
            accuracy = total_correct / total_eval
            print(f"{eval_mode} Accuracy: {accuracy:.2f}")

