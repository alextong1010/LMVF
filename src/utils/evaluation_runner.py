from src.utils.model import ModelRunner
from src.utils.dataset_manager import DatasetManager
from termcolor import colored
from tqdm import trange, tqdm
from src.utils.util import check_correct_answer
from src.utils.mav import MAV

class EvaluationRunner(ModelRunner):
    def __init__(self, config: dict):
        super().__init__(config)
        self.eval_scenarios = config['eval_scenarios']
        self.indiv_pass_at_n_correct_counts = {}
        self.indiv_bon_mav_correct_counts = {model: 0 for model in config['base_config']['models']}
        self.mixed_pass_at_n_correct_count = 0
        self.mixed_bon_mav_correct_count = 0
        self.mg_bon_mav_correct_count = 0
        # self.templates = self.prompt_config['verifier']['templates']

    def run_eval_scenarios(self, datasetManager: DatasetManager):
        for scenario in self.eval_scenarios:
            print(colored(f"Running scenario: {scenario['name']}", "white", attrs=["bold"]))
            self._run_eval_scenario(datasetManager, scenario)

    def _run_eval_scenario(self, datasetManager: DatasetManager, scenario: dict):
        if scenario['mode'] == 'individual':
            if 'pass@' in scenario['eval_mode']:
                self._run_individual_pass_at_n(datasetManager, scenario)
            elif 'bon-mav' in scenario['eval_mode']:
                self._run_individual_bon_mav(datasetManager, scenario)
            else:
                raise ValueError(f"Invalid eval mode: {scenario['eval_mode']}")
        elif scenario['mode'] == 'mixed':
            if 'pass@' in scenario['eval_mode']:
                self._run_mixed_pass_at_n(datasetManager, scenario)
            elif 'bon-mav' in scenario['eval_mode']:
                self._run_mixed_bon_mav(datasetManager, scenario)
            elif 'mg-bon-mav' in scenario['eval_mode']:
                self._run_mg_bon_mav(datasetManager, scenario)
            else:
                raise ValueError(f"Invalid eval mode: {scenario['eval_mode']}")
        else:
            raise ValueError(f"Invalid mode: {scenario['mode']}")

    def _run_individual_pass_at_n(self, datasetManager: DatasetManager, scenario: dict):
        model_list = self._get_model_list(scenario)
        n = int(scenario['eval_mode'].split('@')[1])
        self.indiv_pass_at_n_correct_counts[n] = {model: 0 for model in self.config['base_config']['models']}
        for d in tqdm(datasetManager.dataset, desc=f"Pass@N Eval Task {datasetManager.task_id}"):
            gt_answer = d['gt_answer']
            for model in model_list:
                generated_answers = d[f'generated_answers_{model}']
                for generated_answer in generated_answers[:n]:
                    if check_correct_answer(generated_answer, gt_answer, datasetManager.dataset_name):
                        self.indiv_pass_at_n_correct_counts[n][model] += 1
                        break
        accuracy = {key: value / datasetManager.dataset_size for key, value in self.indiv_pass_at_n_correct_counts[n].items()}
        print(colored(f"Pass@N results: {accuracy}", "white"))
        # return correct_count

    def _run_individual_bon_mav(self, datasetManager: DatasetManager, scenario: dict):
        model_list = self._get_model_list(scenario)
        init_verifier_model = self.config['base_config']['verifier_model']
        strict_verifier_model = self.config['base_config']['strict_verifier_model']
        batch_size = self.config['base_config']['batch_size']
        num_generations = scenario['generations_per_model']
        mav = MAV(datasetManager)
        mav.load_domain_specific_verifiers()
        num_veras = len(mav.veras)
        direct_veras_idx = [i for i, v in enumerate(mav.veras) if "direct" in v]
        # Initial Verification
        init_verifier_model_config = self.all_model_configs[init_verifier_model]
        init_verifier_chat_template = init_verifier_model_config['model']['chat_template']
        init_verifier_llm, init_verifier_sampling_params = self._load_model_and_sampling_params(init_verifier_model_config)
        for model_name in model_list:
            for i in trange(0, int(datasetManager.dataset_size), batch_size, desc=f"Bon-Mav Eval Task {datasetManager.task_id}"):
                batch = datasetManager.dataset[i:i + batch_size]
                problems = self._get_problems(batch)
                solutions = self._get_solutions(batch, model_name, num_generations)
                prompts = mav.create_vera_prompts(problems, solutions)
                user_prompts = [self._to_chat_prompt(init_verifier_chat_template, prompt) for prompt in prompts]
                outputs = self._generate(init_verifier_llm, init_verifier_sampling_params, user_prompts)
                assert len(outputs) == len(batch) * num_veras * num_generations, colored(f"Number of outputs {len(outputs)} is not divisible by the number of veras {num_veras} * batch size {len(batch)} * num_generations {num_generations}", "red")
                self._init_post_processing(batch, outputs, init_verifier_model_config, num_veras, num_generations)
        init_verifier_llm = self._cleanup(init_verifier_llm)

        # Strict Verification
        strict_verifier_model_config = self.all_model_configs[strict_verifier_model]
        strict_verifier_chat_template = strict_verifier_model_config['model']['chat_template']
        strict_verifier_llm, strict_verifier_sampling_params = self._load_model_and_sampling_params(strict_verifier_model_config)
        # breakpoint()
        for model_name in model_list:
            for i in trange(0, int(datasetManager.dataset_size), batch_size, desc=f"Strict Verifications Eval Task {datasetManager.task_id}"):
                batch = datasetManager.dataset[i:i + batch_size]
                problems = self._get_problems(batch)
                solutions = self._get_solutions(batch, model_name, num_generations)
                prompts = mav.create_vera_prompts(problems, solutions)
                user_prompts = [self._to_chat_prompt(strict_verifier_chat_template, prompt) for prompt in prompts]
                assistant_responses = self._get_assistant_responses(batch, init_verifier_model)
                assistant_prompts = [self._to_chat_prompt(strict_verifier_chat_template, response, assistant=True) for response in assistant_responses]
                # breakpoint()
                assert len(user_prompts) == len(assistant_prompts), colored(f"Number of user prompts {len(user_prompts)} and assistant prompts {len(assistant_prompts)} must be equal", "red")
                all_prompts = [user_prompts[i] + assistant_prompts[i] + self._to_chat_prompt(strict_verifier_chat_template, mav.verifier_prompt_config['ask_for_approval_only'].format(answer_symbol=mav.verifier_prompt_config['answer_symbol'])) for i in range(len(user_prompts))]
                outputs = self._generate(strict_verifier_llm, strict_verifier_sampling_params, all_prompts)
                # breakpoint()
                self._strict_post_processing(batch, outputs, strict_verifier_model_config, num_veras, num_generations, direct_veras_idx, init_verifier_model, mav, self.indiv_bon_mav_correct_counts, model_name, datasetManager)
        strict_verifier_llm = self._cleanup(strict_verifier_llm)
        accuracy = {key: value / datasetManager.dataset_size for key, value in self.indiv_bon_mav_correct_counts.items()}
        print(colored(f"Bon-Mav results: {accuracy}", "white"))
        

    def _run_mixed_pass_at_n(self, datasetManager: DatasetManager, scenario: dict):
        model_list = self._get_model_list(scenario)
        n = int(scenario['eval_mode'].split('@')[1])
        assert (len(model_list) * scenario['generations_per_model']) == n, "Number of models * generations per model must be equal to n"
        for d in tqdm(datasetManager.dataset, desc=f"Pass@N Eval Task {datasetManager.task_id}"):
            gt_answer = d['gt_answer']
            generated_answers = self._aggregate_generated_answers(d, model_list, scenario['generations_per_model'])
            for generated_answer in generated_answers:
                if check_correct_answer(generated_answer, gt_answer, datasetManager.dataset_name):
                    self.mixed_pass_at_n_correct_count += 1
                    break
        accuracy = self.mixed_pass_at_n_correct_count / datasetManager.dataset_size
        print(colored(f"Pass@N results: {accuracy}", "white"))
        # return correct_count

    def _run_mixed_bon_mav(self, datasetManager: DatasetManager, scenario: dict):
        pass

    def _run_mg_bon_mav(self, datasetManager: DatasetManager, scenario: dict):
        pass

    def _get_model_list(self, scenario: dict):
        if scenario['model_subset'] == 'all':
            return self.gen_models
        else:
            return scenario['model_subset']
        
    def _aggregate_generated_answers(self, data: dict, model_list: list[str], generations_per_model: int):
        aggregated_answers = []
        for model in model_list:
            aggregated_answers.extend(data[f'generated_answers_{model}'][:generations_per_model])
        return aggregated_answers
    
    def _load_model_and_sampling_params(self, model_config: dict):
        llm, sampling_params = super()._load_model_and_sampling_params(model_config)
        sampling_params.n = 1
        return llm, sampling_params

    def _get_solutions(self, batch: list[dict] | dict, model_name: str, num_generations: int) -> list[list[str]]: 
        is_batch = isinstance(batch, list)
        batch_solutions = [d[f'generated_solutions_{model_name}'][:num_generations] for d in batch] if is_batch else [batch[f'generated_solutions_{model_name}'][:num_generations]]
        return batch_solutions

    def _init_post_processing(self, batch: list[dict], outputs: list[dict], model_config: dict, num_veras: int, num_generations: int):
        model_name = model_config['model']['name']
        is_reasoning = model_config['model']['reasoning']
        gen_per_problem = num_generations * num_veras
        for batch_idx, d in enumerate(batch):
            verifier_responses = [self._process_text(outputs[batch_idx*gen_per_problem+i].outputs[0].text, is_reasoning) for i in range(gen_per_problem)]
            verifier_responses = [verifier_responses[i:i+num_veras] for i in range(0, len(verifier_responses), num_veras)]
            d[f'verifier_responses_{model_name}'] = verifier_responses

    def _strict_post_processing(self, batch: list[dict], outputs: list[dict], model_config: dict, num_veras: int, num_generations: int, direct_veras_idx: list[int], init_verifier_model: str, mav: MAV, correct_counts: dict, base_model_name: str, datasetManager: DatasetManager):
        model_name = model_config['model']['name']
        is_reasoning = model_config['model']['reasoning']
        gen_per_problem = num_generations * num_veras
        need_direct_veras = len(direct_veras_idx) > 0
        # breakpoint()
        for batch_idx, data in enumerate(batch):
            strict_verifier_responses = [self._process_text(outputs[batch_idx*gen_per_problem+i].outputs[0].text, is_reasoning) for i in range(gen_per_problem)]
            strict_verifier_responses = [strict_verifier_responses[i:i+num_veras] for i in range(0, len(strict_verifier_responses), num_veras)]
            data[f'strict_verifier_responses_{model_name}'] = strict_verifier_responses
            if not need_direct_veras:
                data['final_verifier_responses'] = strict_verifier_responses
            else:
                # if need direct veras, get the responses for that specific index from init verifier
                data['final_verifier_responses'] = [
                    data[f'verifier_responses_{init_verifier_model}'][i] if i in direct_veras_idx 
                    else strict_verifier_responses[i]
                    for i in range(len(strict_verifier_responses))
                ]
            # breakpoint()
            verifier_approval_bools = [[self._extract_verifier_approval(output, mav) for output in sublist] for sublist in data['final_verifier_responses']]
            data['final_verifier_approval_bools'] = verifier_approval_bools
            verifier_approval_sums = [sum(sublist) for sublist in verifier_approval_bools]
            best_index = verifier_approval_sums.index(max(verifier_approval_sums))
            best_generated_answer = data[f'generated_answers_{base_model_name}'][best_index]
            gt_answer = data['gt_answer']
            if check_correct_answer(best_generated_answer, gt_answer, datasetManager.dataset_name):
                correct_counts[base_model_name] += 1
            # breakpoint()


    def _get_assistant_responses(self, batch: list[dict], model_name: str):
        assistant_responses = [response for d in batch 
                                for responses in d[f'verifier_responses_{model_name}'] 
                                for response in responses]
        return assistant_responses
    
    def _extract_verifier_approval(self, verifier_response: str, mav: MAV) -> bool:
        """Extract the verifier's approval from the response."""
        # Define possible answer symbols
        answer_symbol = mav.verifier_prompt_config['answer_symbol'].lower()

        # Initialize answer as None
        answer = None

        # Check for each answer symbol
        last_index = verifier_response.lower().rfind(answer_symbol)
        if last_index != -1:
            answer = verifier_response[last_index + len(answer_symbol):].strip()

        if not answer:
            return False
        
        answer = answer.replace("*", "")  # Remove any asterisks (bolding)
        answer = answer.strip().lower()
        if answer == "true" or answer == "true.":
            return True
        elif answer == "false" or answer == "false.":
            return False
        else:
            # Check if 'true' or 'false' is in the first word
            first_word = answer.split()[0]
            if "true" in first_word:
                print(colored(f"\tSuccess. Found 'true' in first_word.lower(): {first_word.lower()}", "magenta"))
                return True
            elif "false" in first_word:
                print(colored(f"\tSuccess. Found 'false' in first_word.lower(): {first_word.lower()}", "magenta"))
                return False
            else:
                return False



