import os
import yaml
from itertools import product
from src.utils.dataset_manager import DatasetManager

class MAV:
    def __init__(self, datasetManager: DatasetManager) -> None:
        self.datasetManager = datasetManager
        self._load_prompt_config()
        self._get_system_str()
        self.answer_symbol = self.verifier_prompt_config['answer_symbol']
        self.all_strategies = {}
        self.all_descriptions = {}
        for vera_name, vera_config in self.verifier_prompt_config['strategies'].items():
            self.all_strategies[vera_name] = vera_config['prompt'].format(answer_symbol=self.answer_symbol)
            self.all_descriptions[vera_name] = vera_config['description']
        self.cached_strategies = None
        self.cached_descriptions = None

    def load_domain_specific_verifiers(self):
        if self.datasetManager.dataset_name in self.datasetManager.dataset_config and "verifiers" in self.datasetManager.dataset_config[self.datasetManager.dataset_name]:
            self.veras = self.datasetManager.dataset_config[self.datasetManager.dataset_name]["verifiers"]
            # Pre-cache strategy strings to avoid repeated dictionary lookups
            self.cached_strategies = [self.all_strategies[vera] for vera in self.veras]
            # Pre-cache descriptions for the selected verifiers
            self.cached_descriptions = {vera: self.all_descriptions[vera] for vera in self.veras}
            # Create a legend of the verifiers
            self.vera_legend = "\n".join([f"{k}: {v}" for k, v in self.cached_descriptions.items()])
        else:
            raise ValueError(f"Domain-specific verifiers not implemented for dataset {self.datasetManager.dataset_name}.")

    def create_vera_prompts(self, problems: list[str], solutions: list[list[str]]):
        if self.cached_strategies is None:
            raise ValueError("Must call load_domain_specific_verifiers() before create_vera_prompts()")
        
        # Pre-format system strings once per problem-solution pair
        system_prompts = []
        for problem, solution in zip(problems, solutions):
            for sol in solution:
                system_prompts.append(self.system_str.format(problem=problem, solution=sol))
        
        # Combine system prompts with cached strategies efficiently
        prompts = []
        for system_prompt in system_prompts:
            prompts.extend([system_prompt + strategy for strategy in self.cached_strategies])
        
        return prompts

    def create_recombination_prompts(self, problems: list[str], solutions: list[list[str]], ver_bools_list: list[list[list[bool]]], merge_strategy: str):
        if self.cached_strategies is None or self.cached_descriptions is None:
            raise ValueError("Must call load_domain_specific_verifiers() before create_recombination_prompts()")

        self.recomb_prompt = self.recomb_prompt_config['strategies'][merge_strategy]['prompt']
        prompts = []
        for problem, solution, ver_bools in zip(problems, solutions, ver_bools_list):
            sol_str = "\n"
            for i, (sol, sol_ver_bools) in enumerate(zip(solution, ver_bools), 1):
                sol_str += f"--- BEGIN SOLUTION {i} ---\n{sol}\nVerifiers:\n"
                for vera_name, vera_bool in zip(self.veras, sol_ver_bools):
                    status = "PASSED" if vera_bool else "FAILED"
                    sol_str += f"- {vera_name}: {status}\n"
                sol_str += f"--- END SOLUTION {i} ---\n\n"
            
            prompts.append(self.recomb_prompt.format(problem=problem, solutions=sol_str, vera_legend=self.vera_legend))
        
        return prompts

    def _load_prompt_config(self):
        prompts_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'prompts.yaml')
        with open(prompts_path, 'r') as f:
            prompt_config = yaml.safe_load(f)
        self.verifier_prompt_config = prompt_config['verifier']
        self.recomb_prompt_config = prompt_config['recombination']

    def _get_system_str(self):
        if 'math' in self.datasetManager.dataset_name:
            self.system_str = self.verifier_prompt_config['system_str_math']
        else:
            raise ValueError(f"System string not implemented for dataset {self.datasetManager.dataset_name} yet.")