import argparse
from prompts.gen_prompts import get_gen_prompt
import torch
from typing import Optional
from utils.math_utils.util import remove_boxed, last_boxed_only_string
from termcolor import colored
from vllm import LLM

def generate_solutions(
    config: dict,
    model_config: dict,
    data: dict | list[dict], 
    model: LLM,
    sampling_params: dict
) -> dict | list[dict]:
    """
    Generate multiple solutions for one or multiple problems.
    
    Args:
        config: Configuration dictionary
        model_config: Model configuration dictionary
        data: A single data dictionary or a list of data dictionaries
        model: The model to use for generation
        sampling_params: Sampling parameters
    Returns:
        If data is a dict: the input dict with a new "solutions" key containing generated solutions
        If data is a list: the input list of dicts, each with a new "solutions" key
    """
    # Handle single problem or list of problems
    is_batch = isinstance(data, list)
    batch_problems = [d["problem"] for d in data] if is_batch else [data["problem"]]
    
    # Create prompts for each problem
    user_prompts = [get_gen_prompt(config['dataset'], p) for p in batch_problems]
    inputs = [ 
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            },
        ] for prompt in user_prompts
    ]
    print(f"\nGenerating {config['num_generations']} different responses for {len(batch_problems)} problem(s)...\n")
    outputs = model.chat(inputs, sampling_params, use_tqdm=False)

    # Assign solutions during grouping
    for i, d in enumerate(data):
        d['gt_answer'] = extract_answer(d['solution'], config['dataset'])
        if model_config.get("model", {}).get("reasoning"):
            d['generated_solutions'] = []
            for j in range(sampling_params.n):
                try:
                    # Try to split using </think> token
                    solution = (outputs[i].outputs[j].text).split("</think>")[1].split("<end_of_turn>")[0]
                except IndexError:
                    # Default to splitting using <end_of_turn> if </think> is not found
                    solution = (outputs[i].outputs[j].text).split("<end_of_turn>")[0]
                d['generated_solutions'].append(solution)
        else:
            d['generated_solutions'] = [(outputs[i].outputs[j].text).split("<end_of_turn>")[0] for j in range(sampling_params.n)]
        d['generated_answers'] = [extract_answer(solution, config['dataset']) for solution in d['generated_solutions']]
    return data if is_batch else data[0]


def extract_answer(solution: str, dataset_name: str, err_msg: Optional[str] = None) -> str:
    """Extract the answer from the solution."""
    if dataset_name == "math":
        # use provided extraction function
        answer = remove_boxed(last_boxed_only_string(solution))
        answer = answer.replace("**", "")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if not answer:
        # Answer is None or empty string
        if isinstance(answer, str) and len(answer) == 0:
            # Is empty string, check if '\\boxed{}' is present (if present, extracted answer is empty string)
            if "\\boxed{}" in solution:
                return ""
        print(
            colored(f"\nWARNING in extract_answer, found no answer: {answer=} with {type(answer)=} ({dataset_name=}), "
                    f"and full solution (length {len(solution)}) is: \n{'-' * 30}\n{solution}\n{'-' * 30} (WARNING in extract_answer)\n"
                    f"{('     ERROR MESSAGE: ' + err_msg) if err_msg is not None else ''}", "yellow"))
        return None

    return answer