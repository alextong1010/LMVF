from src.utils.model import ModelRunner
from src.utils.dataset_manager import DatasetManager
from termcolor import colored
from tqdm import trange, tqdm
from src.utils.util import check_correct_answer, extract_answer
from src.utils.mav import MAV
import os
import yaml

MANUAL_OFFSET = 6

class EvaluationRunner(ModelRunner):
    def __init__(self, config: dict, datasetManager: DatasetManager, output_dirpath: str):
        super().__init__(config, output_dirpath)
        self.eval_scenarios = config['eval_scenarios']
        self.mav = MAV(datasetManager)
        self.mav.load_domain_specific_verifiers()
        self.num_veras = len(self.mav.veras)

    def run_eval_scenarios(self):
        for scenario in self.eval_scenarios:
            print(colored(f"Running scenario: {scenario['name']}", "white", attrs=["bold"]))
            self._run_eval_scenario(scenario)
    
    def save_results(self):
        # Saves the config (which now contains the accuracy) and the dataset
        with open(os.path.join(self.output_dirpath, 'eval_config.yaml'), 'w') as f:
            yaml.dump(self.config, f)
        self.mav.datasetManager.save_dataset(os.path.join(self.output_dirpath, 'eval_outputs.json'))

    def _run_eval_scenario(self, scenario: dict):
        if scenario['mode'] == 'individual':
            if 'pass@' in scenario['eval_mode']:
                self._run_individual_pass_at_n(scenario)
            elif 'bon-mav' in scenario['eval_mode']:
                self._run_individual_bon_mav(scenario)
            else:
                raise ValueError(f"Invalid eval mode: {scenario['eval_mode']}")
        elif scenario['mode'] == 'mixed':
            if 'pass@' in scenario['eval_mode']:
                self._run_mixed_pass_at_n(scenario)
            elif 'mg-bon-mav' in scenario['eval_mode']:
                self._run_mg_bon_mav(scenario)
            elif 'bon-mav' in scenario['eval_mode']:
                self._run_mixed_bon_mav(scenario)
            else:
                raise ValueError(f"Invalid eval mode: {scenario['eval_mode']}")
        else:
            raise ValueError(f"Invalid mode: {scenario['mode']}")

    def _run_individual_pass_at_n(self, scenario: dict):
        model_list = self._get_model_list(scenario)
        n = int(scenario['eval_mode'].split('@')[1])
        indiv_pass_at_n_correct_counts = {model: 0 for model in model_list}
        for d in tqdm(self.mav.datasetManager.dataset, desc=f"Pass@N Eval Task {self.mav.datasetManager.task_id}"):
            gt_answer = d['gt_answer']
            for model in model_list:
                generated_answers = d[f'generated_answers_{model}']
                for generated_answer in generated_answers[MANUAL_OFFSET:n+MANUAL_OFFSET]:
                    if check_correct_answer(generated_answer, gt_answer, self.mav.datasetManager.dataset_name):
                        indiv_pass_at_n_correct_counts[model] += 1
                        break
        accuracy = {key: value / self.mav.datasetManager.dataset_size for key, value in indiv_pass_at_n_correct_counts.items()}
        print(colored(f"Pass@N results: {accuracy}", "white"))
        scenario['accuracy'] = accuracy

    def _run_individual_bon_mav(self, scenario: dict):
        model_list = self._get_model_list(scenario)
        init_verifier_model = scenario['verifier_model']
        strict_verifier_model = scenario['strict_verifier_model']
        num_generations = scenario['generations_per_model']
        indiv_bon_mav_correct_counts = {model: 0 for model in model_list}
        print(f"Using init verifier model: {init_verifier_model}")
        print(f"Using strict verifier model: {strict_verifier_model}")
        print(f"Using num generations: {num_generations}")
        for d in tqdm(self.mav.datasetManager.dataset, desc=f"Bon-Mav Eval Task {self.mav.datasetManager.task_id}"):
            gt_answer = d['gt_answer']
            for model_name in model_list:
                verifier_approval_sums = d[f'fin_ver_approval_sums_{strict_verifier_model}_on_ver_{init_verifier_model}_on_{model_name}'][MANUAL_OFFSET:num_generations+MANUAL_OFFSET]
                best_index = verifier_approval_sums.index(max(verifier_approval_sums))
                best_generated_answer = d[f'generated_answers_{model_name}'][best_index + MANUAL_OFFSET]
                if check_correct_answer(best_generated_answer, gt_answer, self.mav.datasetManager.dataset_name):
                    indiv_bon_mav_correct_counts[model_name] += 1
        accuracy = {key: value / self.mav.datasetManager.dataset_size for key, value in indiv_bon_mav_correct_counts.items()}
        print(colored(f"Bon-Mav results: {accuracy}", "white"))
        scenario['accuracy'] = accuracy

    def _run_mixed_pass_at_n(self, scenario: dict):
        model_list = self._get_model_list(scenario)
        n = int(scenario['eval_mode'].split('@')[1])
        num_generations = scenario['generations_per_model']
        assert (len(model_list) * num_generations) == n, "Number of models * generations per model must be equal to n"
        mixed_pass_at_n_correct_count = 0
        for d in tqdm(self.mav.datasetManager.dataset, desc=f"Pass@N Eval Task {self.mav.datasetManager.task_id}"):
            gt_answer = d['gt_answer']
            generated_answers = self._aggregate_generated_answers(d, model_list, num_generations)
            for generated_answer in generated_answers:
                if check_correct_answer(generated_answer, gt_answer, self.mav.datasetManager.dataset_name):
                    mixed_pass_at_n_correct_count += 1
                    break
        accuracy = mixed_pass_at_n_correct_count / self.mav.datasetManager.dataset_size
        print(colored(f"Pass@N results: {accuracy}", "white"))
        scenario['accuracy'] = accuracy

    def _run_mixed_bon_mav(self, scenario: dict):
        model_list = self._get_model_list(scenario)
        init_verifier_model = scenario['verifier_model']
        strict_verifier_model = scenario['strict_verifier_model']
        num_generations = scenario['generations_per_model']
        mixed_bon_mav_correct_count = 0
        
        for d in tqdm(self.mav.datasetManager.dataset, desc=f"Mixed Bon-Mav Eval Task {self.mav.datasetManager.task_id}"):
            gt_answer = d['gt_answer']
            all_verifier_approval_sums = []
            all_generated_answers = []
            
            for model_name in model_list:
                verifier_approval_sums = d[f'fin_ver_approval_sums_{strict_verifier_model}_on_ver_{init_verifier_model}_on_{model_name}'][MANUAL_OFFSET:num_generations+MANUAL_OFFSET]
                generated_answers = d[f'generated_answers_{model_name}'][MANUAL_OFFSET:num_generations+MANUAL_OFFSET]
                
                all_verifier_approval_sums.extend(verifier_approval_sums)
                all_generated_answers.extend(generated_answers)
            
            # Find the best answer across all models based on verifier scores
            best_index = all_verifier_approval_sums.index(max(all_verifier_approval_sums))
            best_generated_answer = all_generated_answers[best_index]
            
            if check_correct_answer(best_generated_answer, gt_answer, self.mav.datasetManager.dataset_name):
                mixed_bon_mav_correct_count += 1
        
        accuracy = mixed_bon_mav_correct_count / self.mav.datasetManager.dataset_size
        print(colored(f"Mixed Bon-Mav results: {accuracy}", "white"))
        scenario['accuracy'] = accuracy

    def _run_mg_bon_mav(self, scenario: dict):
        model_list = self._get_model_list(scenario)
        init_verifier_model = scenario['verifier_model']
        strict_verifier_model = scenario['strict_verifier_model']
        recomb_model = scenario['recomb_model']
        num_generations = scenario['generations_per_model']
        merge_strategy = scenario['merge_strategy']
        mg_bon_mav_correct_count = 0
        self._recombine(model_list, init_verifier_model, strict_verifier_model, recomb_model, num_generations, merge_strategy)
        for d in tqdm(self.mav.datasetManager.dataset, desc=f"MG Bon-Mav Eval Task {self.mav.datasetManager.task_id}"):
            gt_answer = d['gt_answer']
            recombined_answer = d[f'recombined_answer']
            if check_correct_answer(recombined_answer, gt_answer, self.mav.datasetManager.dataset_name):
                mg_bon_mav_correct_count += 1  
        accuracy = mg_bon_mav_correct_count / self.mav.datasetManager.dataset_size
        print(colored(f"MG Bon-Mav results: {accuracy}", "white"))
        scenario['accuracy'] = accuracy

    def _recombine(self, model_list: list[str], init_verifier_model: str, strict_verifier_model: str, recomb_model: str, num_generations: int, merge_strategy: str):
        recomb_model_config = self.all_model_configs[recomb_model]
        recomb_chat_template = recomb_model_config['model']['chat_template']
        recomb_llm, recomb_sampling_params = self._load_model_and_sampling_params(recomb_model_config)
        recomb_sampling_params.n = 1
        batch_size = self.config['base_config']['batch_size']
        is_reasoning = recomb_model_config['model']['reasoning']
        dataset_name = self.mav.datasetManager.dataset_name
        for i in trange(0, self.mav.datasetManager.dataset_size, batch_size, desc=f"Recombining Task {self.mav.datasetManager.task_id}"):
            batch = self.mav.datasetManager.dataset[i:i+batch_size]
            problems = self._get_problems(batch)
            solutions = self._get_solutions(batch, model_list, num_generations)
            ver_bools_list = self._get_ver_bools_list(batch, model_list, num_generations, strict_verifier_model, init_verifier_model)
            prompts = self.mav.create_recombination_prompts(problems, solutions, ver_bools_list, merge_strategy)
            user_prompts = [self._to_chat_prompt(recomb_chat_template, prompt) for prompt in prompts]
            if len(user_prompts) != len(batch):
                raise AssertionError(colored(f"Number of user prompts {len(user_prompts)} is not equal to batch size {len(batch)}", "red"))
            outputs = self._generate(recomb_llm, recomb_sampling_params, user_prompts)
            for d, output in zip(batch, outputs):
                d['recombined_solution'] = self._process_text(output.outputs[0].text, is_reasoning)
                d['recombined_answer'] = extract_answer(d['recombined_solution'], dataset_name)

    def _get_solutions(self, batch: list[dict], model_list: list[str], num_generations: int) -> list[list[str]]:
        batch_solutions = []
        for d in batch:
            solutions = []
            for model_name in model_list:
                solutions.extend(d[f'generated_solutions_{model_name}'][MANUAL_OFFSET:num_generations+MANUAL_OFFSET])
            batch_solutions.append(solutions)
        return batch_solutions
    
    def _get_ver_bools_list(self, batch: list[dict], model_list: list[str], num_generations: int, strict_verifier_model: str, init_verifier_model: str) -> list[list[list[bool]]]:
        batch_ver_bools_list = []
        for d in batch:
            ver_bools = []
            for model_name in model_list:
                ver_bools.extend(d[f'fin_ver_approval_bools_{strict_verifier_model}_on_ver_{init_verifier_model}_on_{model_name}'][MANUAL_OFFSET:num_generations+MANUAL_OFFSET])
            batch_ver_bools_list.append(ver_bools)
        return batch_ver_bools_list

    def _get_model_list(self, scenario: dict):
        if scenario['model_subset'] == 'all':
            return self.gen_models
        else:
            return scenario['model_subset']

    def _aggregate_generated_answers(self, data: dict, model_list: list[str], num_generations: int):
        aggregated_answers = []
        for model in model_list:
            aggregated_answers.extend(data[f'generated_answers_{model}'][MANUAL_OFFSET:num_generations+MANUAL_OFFSET])
        return aggregated_answers



