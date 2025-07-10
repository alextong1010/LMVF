import os
import yaml
from itertools import product
from src.utils.dataset_manager import DatasetManager

class MAV:
    def __init__(self, datasetManager: DatasetManager) -> None:
        self.datasetManager = datasetManager
        self._load_prompt_config()

    def load_domain_specific_verifiers(self):
        if self.datasetManager.dataset_name in self.datasetManager.dataset_config and "verifiers" in self.datasetManager.dataset_config[self.datasetManager.dataset_name]:
            self.veras = self.datasetManager.dataset_config[self.datasetManager.dataset_name]["verifiers"]
        else:
            raise ValueError(f"Domain-specific verifiers not implemented for dataset {self.datasetManager.dataset_name}.")

    def create_vera_prompts(self, problems: list[str], solutions: list[list[str]]):
        prompts = [
            self.verifier_prompt_config['system_str_math'].format(problem=problem, solution=sol) + 
            self.verifier_prompt_config['strategies'][vera]['prompt'].format(answer_symbol=self.verifier_prompt_config['answer_symbol'])
            for problem, solution in zip(problems, solutions)
            for sol, vera in product(solution, self.veras)
        ]
        return prompts

    def create_recombination_prompts(self):
        pass

    def _load_prompt_config(self):
        prompts_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'prompts.yaml')
        with open(prompts_path, 'r') as f:
            prompt_config = yaml.safe_load(f)
        self.verifier_prompt_config = prompt_config['verifier']