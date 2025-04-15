from utils.dataset_utils import check_correct_answer
from utils.vera_utils import generate_verifier_approvals
from tqdm import trange
from datetime import datetime

def evaluate_problem(config, data_with_generated_solutions, verifier_llm, verifier_sampling_params, strict_verifier_sampling_params):
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
            correct_count = evaluate_problem_bon_mav(config, data_with_generated_solutions, verifier_llm, verifier_sampling_params, strict_verifier_sampling_params)
        else:
            raise ValueError(f"Invalid eval_mode(s): {eval_mode}")
        results[eval_mode] = correct_count
        
        eval_mode_end_time = datetime.now()
        print(f"Evaluation of {eval_mode} took: {eval_mode_end_time - eval_mode_start_time}")
    return results, data_with_generated_solutions

def evaluate_problem_bon_mav(config, data_with_generated_solutions, verifier_llm, verifier_sampling_params, strict_verifier_sampling_params):
    """
    Evaluates the model on a single problem or batch of problems using the bon-mav evaluation metric.
    
    Args:
        config (dict): The configuration.
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
            config, batch, verifier_llm, verifier_sampling_params, strict_verifier_sampling_params
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


