import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.gen_utils import extract_answer
from typing import List
import json
from functools import lru_cache
@lru_cache(maxsize=4096)
def _cached_extract(ans, dataset):
    return extract_answer(ans, dataset)

class SanityCheck:
    def __init__(self, config, prompt_version):
        self.config = config
        self.__name__ = "sanity_check"
        self.prompt_version = prompt_version

    def __call__(self,
                 prompts:     List[str],
                 completions: List[str],
                 problem: List[str],
                 level: List[str],
                 type: List[str],
                 solution: List[str],
                 clean_solution: List[str],
                 **kwargs) -> List[float]:
        if self.prompt_version == 1:
            answers = [c[0]['content'][0]['text'] for c in completions]
        elif self.prompt_version == 2:
            answers = [c[0]['content'] for c in completions]
        elif self.prompt_version == 3:
            answers = completions
        ans  = [_cached_extract(a, self.config['dataset']) for a in answers]
        result = [int(x == y) for x, y in zip(ans, clean_solution)]
        
        scaled_result = [(r / 6) ** 2 for r in result]
        n = len(result)
        rank_bonus = [0.05 * (1 - i / (n - 1)) if n > 1 else 0.05 for i in range(n)]

        result_shaped = [b + r for b, r in zip(scaled_result, rank_bonus)]

        # save only the result to file and make sure it doesn't overwrite the previous result
        # add new line between each result
        result_str = "\n".join(map(str, result))
        result = {
            "clean_solution": clean_solution,
            "answers": ans,
            "result": result_str,
            "result_shaped": result_shaped
        }
        if not os.path.exists(f"sanity_check_result.jsonl"):
            with open(f"sanity_check_result.jsonl", "w") as f:
                f.write(json.dumps(result) + "\n")
        else:
            with open(f"sanity_check_result.jsonl", "a") as f:
                f.write(json.dumps(result) + "\n")

        return result_shaped