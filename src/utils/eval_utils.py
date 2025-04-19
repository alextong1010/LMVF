from utils.dataset_utils import check_correct_answer
from utils.vera_utils import generate_initial_verifier_responses, generate_strict_verifier_approvals
from tqdm import trange
from datetime import datetime
from vllm import LLM
from typing import List, Dict, Optional, Any

def evaluate_problem(config, verifier_model_config, data_with_generated_solutions, verifier_llm, verifier_sampling_params, strict_verifier_sampling_params):
    """
    Evaluates the model on a single problem or batch of problems using the pass@n or bon-mav evaluation metric.

    Args:
        config (dict): The configuration.
        data_with_generated_solutions (list): List of dictionaries containing "generated_answers" and "gt_answer".
        verifier_llm: The language model to use for verification.
        verifier_sampling_params: The sampling parameters for the verifier model.
        strict_verifier_sampling_params: The sampling parameters for the strict verifier model.

    Returns:
        dict: A dictionary where keys are eval modes and values are the number of correct predictions.
    """
    # Handle the case where eval_mode is a list
    eval_modes = config['eval_mode'] if isinstance(config['eval_mode'], list) else [config['eval_mode']]
    
    results = {}
    for eval_mode in eval_modes:
        eval_mode_start_time = datetime.now()
        print(f"Evaluating {eval_mode}...")
        if eval_mode.startswith("pass@"):
            try:
                n = int(eval_mode.split("@")[1])
                correct_count = evaluate_problem_pass_at_n(config, data_with_generated_solutions, n)
            except (ValueError, IndexError):
                raise ValueError(f"Invalid eval_mode(s) format: {eval_mode}. Expected format: pass@n where n is a number.")
        elif eval_mode == "bon-mav": # BoN-MAV runs through all of the generated answers
            correct_count = evaluate_problem_bon_mav(config, verifier_model_config, data_with_generated_solutions, verifier_llm, verifier_sampling_params, strict_verifier_sampling_params)
        else:
            raise ValueError(f"Invalid eval_mode(s): {eval_mode}")
        results[eval_mode] = correct_count
        
        eval_mode_end_time = datetime.now()
        print(f"Evaluation of {eval_mode} took: {eval_mode_end_time - eval_mode_start_time}")
    return results, data_with_generated_solutions

def evaluate_problem_bon_mav(config, verifier_model_config, data_with_generated_solutions, verifier_llm, verifier_sampling_params, strict_verifier_sampling_params):
    """
    Evaluates the model on a single problem or batch of problems using the bon-mav evaluation metric.
    
    Args:
        config (dict): The configuration.
        verifier_model_config (dict): The configuration for the verifier model.
        data_with_generated_solutions (list): List of dictionaries containing generated solutions.
        verifier_llm: The language model to use for verification.
        verifier_sampling_params: The sampling parameters for the verifier model.
        strict_verifier_sampling_params: The sampling parameters for the strict verifier model.
    """
    correct_count = 0 
    batch_size = config['verifier_batch_size']
    
    # Process data in batches
    for i in trange(0, len(data_with_generated_solutions), batch_size, desc="Verifying"):
        batch = data_with_generated_solutions[i:i + batch_size]
        batch_data = generate_verifier_approvals(
            config, verifier_model_config, batch, verifier_llm, verifier_sampling_params, strict_verifier_sampling_params
        )

        for data in batch_data:
            gt_answer = data['gt_answer']
            generated_answers = data['generated_answers']
            verifier_approval_bools = data['verifier_approval_bools']
            approval_sums = [sum(sublist) for sublist in verifier_approval_bools]
            best_index = approval_sums.index(max(approval_sums))
            best_generated_answer = generated_answers[best_index]

            if config['verbose']:
                print(f"gt_answer: {gt_answer}")
                print(f"generated_answers: {generated_answers}")
                print(f"verifier_approval_bools: {verifier_approval_bools}")
                print(f"best_index: {best_index}")
                print(f"best_generated_answer: {best_generated_answer}")

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

def run_initial_verification_batch(
    config: dict,
    verifier_model_config: dict,
    batch_data: List[Dict[str, Any]],
    verifier_llm: LLM,
    verifier_sampling_params: dict,
) -> List[Dict[str, Any]]:
    """
    Runs the initial verification pass using the verifier LLM.

    Args:
        config: Configuration dictionary.
        verifier_model_config: Configuration for the verifier model.
        batch_data: List of dictionaries containing generated solutions.
        verifier_llm: The language model for initial verification.
        verifier_sampling_params: Sampling parameters for the verifier model.

    Returns:
        list: Updated batch data with 'verifier_responses'.
    """
    print(f"Running initial verification for batch of size {len(batch_data)}...")
    # Use the new vera_utils function for the first pass
    updated_batch_data = generate_initial_verifier_responses(
        config, verifier_model_config, batch_data, verifier_llm, verifier_sampling_params
    )
    print("Initial verification complete for batch.")
    return updated_batch_data

def run_strict_verification_and_evaluate_batch(
    config: dict,
    strict_verifier_model_config: dict,
    batch_data: List[Dict[str, Any]],
    strict_verifier_llm: Optional[LLM], # Optional for skip_llm_call
    strict_verifier_sampling_params: Optional[dict], # Optional for skip_llm_call
    skip_llm_call: bool = False # Flag to skip LLM call if data already processed
) -> Dict[str, Any]:
    """
    Runs the strict verification pass and evaluates BoN-MAV correctness.

    Args:
        config: Configuration dictionary.
        strict_verifier_model_config: Configuration for the strict verifier model.
        batch_data: List of dictionaries containing generated solutions and 'verifier_responses'.
        strict_verifier_llm: The language model for strict verification. Can be None if skip_llm_call is True.
        strict_verifier_sampling_params: Sampling parameters for strict verifier. Can be None if skip_llm_call is True.
        skip_llm_call: If True, skips the LLM call and assumes final approvals exist in batch_data.

    Returns:
        dict: Contains 'correct_count' for the batch and 'updated_batch_data' with final approvals.
    """
    correct_count = 0
    
    if not skip_llm_call:
        if strict_verifier_llm is None or strict_verifier_sampling_params is None:
             raise ValueError("strict_verifier_llm and strict_verifier_sampling_params must be provided if skip_llm_call is False.")
        print(f"Running strict verification for batch of size {len(batch_data)}...")
        # Use the new vera_utils function for the second pass
        batch_data = generate_strict_verifier_approvals(
            config, strict_verifier_model_config, batch_data, strict_verifier_llm, strict_verifier_sampling_params
        )
        print("Strict verification complete for batch.")
    else:
        print(f"Skipping strict verification LLM call for batch of size {len(batch_data)}.")
        # Ensure required keys exist if skipping LLM call
        if not batch_data or 'verifier_approval_bools' not in batch_data[0]:
             raise ValueError("Cannot skip LLM call: 'verifier_approval_bools' not found in batch data.")


    # Evaluate correctness based on the final approvals
    print(f"Evaluating BoN-MAV correctness for batch of size {len(batch_data)}...")
    for data in batch_data:
        gt_answer = data['gt_answer']
        generated_answers = data['generated_answers'] # Assuming 'generated_answers' key exists
        verifier_approval_bools = data['verifier_approval_bools']
        
        # Check if verifier_approval_bools has the expected structure
        if not isinstance(verifier_approval_bools, list) or not all(isinstance(sublist, list) for sublist in verifier_approval_bools):
             print(f"Warning: Unexpected structure for verifier_approval_bools: {verifier_approval_bools}. Skipping evaluation for this item.")
             continue
        if len(verifier_approval_bools) != len(generated_answers):
             print(f"Warning: Mismatch between number of generated answers ({len(generated_answers)}) and approval lists ({len(verifier_approval_bools)}). Skipping evaluation for this item.")
             continue

        approval_sums = [sum(sublist) for sublist in verifier_approval_bools]

        # Handle cases where no approvals were generated (e.g., errors upstream)
        if not approval_sums:
             print(f"Warning: No approval sums generated for data item. Skipping evaluation.")
             continue

        best_index = approval_sums.index(max(approval_sums))
        best_generated_answer = generated_answers[best_index]

        if config['verbose']:
            print(f"gt_answer: {gt_answer}")
            # print(f"generated_answers: {generated_answers}") # Can be verbose
            print(f"verifier_approval_bools: {verifier_approval_bools}")
            print(f"approval_sums: {approval_sums}")
            print(f"best_index: {best_index}")
            print(f"best_generated_answer: {best_generated_answer}")

        is_correct = check_correct_answer(best_generated_answer, gt_answer, config['dataset'])
        if is_correct:
            correct_count += 1

    print(f"BoN-MAV evaluation complete for batch. Correct: {correct_count}/{len(batch_data)}")
    return {'correct_count': correct_count, 'updated_batch_data': batch_data}


