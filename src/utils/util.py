from src.utils.math_utils.util import remove_boxed, last_boxed_only_string
from termcolor import colored
from typing import Optional
from src.utils.math_utils.math_equivalence import is_equiv



def check_correct_answer(answer, correct_answer, dataset_name):
    if dataset_name == "math_500":
        return is_equiv(answer, correct_answer)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def extract_answer(solution: str, dataset_name: str, err_msg: Optional[str] = None) -> str:
    """Extract the answer from the solution."""
    if dataset_name == "math_500":
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
        # print(
        #     colored(f"\nWARNING in extract_answer, found no answer: {answer=} with {type(answer)=} ({dataset_name=}), "
        #             f"and full solution (length {len(solution)}) is: \n{'-' * 30}\n{solution}\n{'-' * 30} (WARNING in extract_answer)\n"
        #             f"{('     ERROR MESSAGE: ' + err_msg) if err_msg is not None else ''}", "yellow"))
        return None

    return answer

def assert_dataset_contains_req_models(dataset_manager, models):
    for model in models:
        if f'generated_solutions_{model}' not in dataset_manager.dataset[0].keys() or \
            f'generated_answers_{model}' not in dataset_manager.dataset[0].keys():
            raise ValueError(f"Model {model} not found in dataset")



