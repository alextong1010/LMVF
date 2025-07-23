from src.utils.model import ModelRunner
from tqdm import trange
from src.utils.dataset_manager import DatasetManager
from src.utils.mav import MAV
from termcolor import colored
import os
import yaml

class VerifierRunner(ModelRunner):
    def __init__(self, config: dict, datasetManager: DatasetManager, output_dirpath: str):
        super().__init__(config, output_dirpath)

        # Load verifier and strict verifier models
        self.ver_models = self.config['base_config']['verifier_models']
        if isinstance(self.ver_models, str):
            self.ver_models = [self.ver_models]
        self.strict_ver_models = self.config['base_config']['strict_verifier_models']
        if isinstance(self.strict_ver_models, str):
            self.strict_ver_models = [self.strict_ver_models]

        self.ver_models_configs = {model: self.all_model_configs[model] for model in self.ver_models}
        self.strict_ver_models_configs = {model: self.all_model_configs[model] for model in self.strict_ver_models}

        # Load other configs
        self.batch_size = self.config['base_config']['batch_size']
        self.num_generations = self.config['base_config']['num_generations']

        self.mav = MAV(datasetManager)
        self.mav.load_domain_specific_verifiers()
        self.num_veras = len(self.mav.veras)
        self.direct_veras_idx = [i for i, v in enumerate(self.mav.veras) if "direct" in v]
        self.solutions_file_name = self.config['base_config']['solutions_file_name'].format(task_id=self.mav.datasetManager.task_id)

    def save_verifications(self):
        with open(os.path.join(self.output_dirpath, 'ver_config.yaml'), 'w') as f:
            yaml.dump(self.config, f)
        self.mav.datasetManager.save_dataset(os.path.join(self.output_dirpath, self.solutions_file_name))

    def generate_init_verifications(self):
        # Pre-compute prompts once for all verifier models
        self._precompute_prompts()
        
        # Iterate over each verifier model
        for ver_model in self.ver_models:
            ver_model_config = self.ver_models_configs[ver_model]
            ver_model_chat_template = ver_model_config['model']['chat_template']
            ver_model_llm, ver_model_sampling_params = self._load_model_and_sampling_params(ver_model_config)
            ver_model_sampling_params.n = 1
            for gen_model in self.gen_models:
                ver_resp = f'ver_resp_{ver_model}_on_{gen_model}'
                for i in trange(0, self.mav.datasetManager.dataset_size, self.batch_size, desc=f"Verifying using {ver_model} on {gen_model} Task {self.mav.datasetManager.task_id}"):
                    batch = self.mav.datasetManager.dataset[i:i + self.batch_size]
                    prompts = self.cached_prompts[gen_model][i]
                    user_prompts = [self._to_chat_prompt(ver_model_chat_template, prompt) for prompt in prompts]
                    if len(user_prompts) != len(batch) * self.num_veras * self.num_generations:
                        raise AssertionError(colored(f"Number of user prompts {len(user_prompts)} is not divisible by the number of veras {self.num_veras} * batch size {len(batch)} * num_generations {self.num_generations}", "red"))
                    outputs = self._generate(ver_model_llm, ver_model_sampling_params, user_prompts)
                    self._init_post_processing(batch, outputs, ver_model_config, gen_model, self.num_veras, self.num_generations, ver_resp)
            ver_model_llm = self._cleanup(ver_model_llm)

    def generate_strict_verifications(self):
        for strict_ver_model in self.strict_ver_models:
            strict_ver_model_config = self.strict_ver_models_configs[strict_ver_model]
            strict_ver_model_chat_template = strict_ver_model_config['model']['chat_template']
            strict_ver_model_llm, strict_ver_model_sampling_params = self._load_model_and_sampling_params(strict_ver_model_config)
            strict_ver_model_sampling_params.n = 1
            # Pre-compute the strict prompt once
            strict_prompt = self._to_chat_prompt(strict_ver_model_chat_template, 
                                                 self.mav.verifier_prompt_config['ask_for_approval_only'].format(
                                                     answer_symbol=self.mav.verifier_prompt_config['answer_symbol']))
            
            for ver_model in self.ver_models:
                for gen_model in self.gen_models:
                    ver_resp, strict_ver_resp, fin_strict_ver_resp, fin_ver_approval_bools, fin_ver_approval_sums = self._get_dict_names(ver_model, gen_model, strict_ver_model)
                    for i in trange(0, self.mav.datasetManager.dataset_size, self.batch_size, desc=f"Running Strict Verifier using {strict_ver_model} on verifier {ver_model} on {gen_model} Task {self.mav.datasetManager.task_id}"):
                        batch = self.mav.datasetManager.dataset[i:i + self.batch_size]
                        # Use cached prompts instead of recomputing
                        prompts = self.cached_prompts[gen_model][i]
                        user_prompts = [self._to_chat_prompt(strict_ver_model_chat_template, prompt) for prompt in prompts]
                        assert len(user_prompts) == len(batch) * self.num_veras * self.num_generations, colored(f"Number of user prompts {len(user_prompts)} is not divisible by the number of veras {self.num_veras} * batch size {len(batch)} * num_generations {self.num_generations}", "red")
                        assistant_responses = self._get_assistant_responses(batch, ver_model, gen_model)
                        assistant_prompts = [self._to_chat_prompt(strict_ver_model_chat_template, response, assistant=True) for response in assistant_responses]
                        all_prompts = [user_prompts[i] + assistant_prompts[i] + strict_prompt for i in range(len(user_prompts))]
                        outputs = self._generate(strict_ver_model_llm, strict_ver_model_sampling_params, all_prompts)
                        self._strict_post_processing(batch, outputs, strict_ver_model_config, ver_model, gen_model, self.num_veras, self.num_generations, self.direct_veras_idx, self.mav, ver_resp, strict_ver_resp, fin_strict_ver_resp, fin_ver_approval_bools, fin_ver_approval_sums)
            strict_ver_model_llm = self._cleanup(strict_ver_model_llm)
        
        # Clean up cached prompts if they exist
        if hasattr(self, 'cached_prompts'):
            del self.cached_prompts

    def _get_solutions(self, batch: list[dict] | dict, model_name: str, num_generations: int) -> list[list[str]]: 
        is_batch = isinstance(batch, list)
        batch_solutions = [d[f'generated_solutions_{model_name}'][:num_generations] for d in batch] if is_batch else [batch[f'generated_solutions_{model_name}'][:num_generations]]
        return batch_solutions
    
    def _precompute_prompts(self):
        self.cached_prompts = {}
        
        for gen_model in self.gen_models:
            self.cached_prompts[gen_model] = {}
            for i in range(0, self.mav.datasetManager.dataset_size, self.batch_size):
                batch = self.mav.datasetManager.dataset[i:i + self.batch_size]
                problems = self._get_problems(batch)
                solutions = self._get_solutions(batch, gen_model, self.num_generations)
                prompts = self.mav.create_vera_prompts(problems, solutions)
                self.cached_prompts[gen_model][i] = prompts
    
    def _init_post_processing(self, batch: list[dict], outputs: list[dict], ver_model_config: dict, gen_model_name: str, num_veras: int, num_generations: int, ver_resp: str):
        is_reasoning = ver_model_config['model']['reasoning']
        gen_per_problem = num_generations * num_veras
        for batch_idx, d in enumerate(batch):
            verifier_responses = [self._process_text(outputs[batch_idx*gen_per_problem+i].outputs[0].text, is_reasoning) for i in range(gen_per_problem)]
            verifier_responses = [verifier_responses[i:i+num_veras] for i in range(0, len(verifier_responses), num_veras)]
            d[ver_resp] = verifier_responses

    def _strict_post_processing(self, batch: list[dict], outputs: list[dict], strict_ver_model_config: dict, ver_model_name: str, gen_model_name: str, num_veras: int, num_generations: int, direct_veras_idx: list[int], mav: MAV, ver_resp: str, strict_ver_resp: str, fin_strict_ver_resp: str, fin_ver_approval_bools: str, fin_ver_approval_sums: str):
        is_reasoning = strict_ver_model_config['model']['reasoning']
        gen_per_problem = num_generations * num_veras
        need_direct_veras = len(direct_veras_idx) > 0

        for batch_idx, d in enumerate(batch):
            strict_verifier_responses = [self._process_text(outputs[batch_idx*gen_per_problem+i].outputs[0].text, is_reasoning) for i in range(gen_per_problem)]
            strict_verifier_responses = [strict_verifier_responses[i:i+num_veras] for i in range(0, len(strict_verifier_responses), num_veras)]
            d[strict_ver_resp] = strict_verifier_responses
            if not need_direct_veras:
                d[fin_strict_ver_resp] = strict_verifier_responses
            else:
                d[fin_strict_ver_resp] = [
                    d[ver_resp][i] if i in direct_veras_idx 
                    else strict_verifier_responses[i]
                    for i in range(len(strict_verifier_responses))
                ]
            verifier_approval_bools = [[self._extract_verifier_approval(output, mav) for output in sublist] for sublist in d[fin_strict_ver_resp]]
            d[fin_ver_approval_bools] = verifier_approval_bools
            verifier_approval_sums = [sum(sublist) for sublist in verifier_approval_bools]
            d[fin_ver_approval_sums] = verifier_approval_sums

    def _get_assistant_responses(self, batch: list[dict], ver_model_name: str, gen_model_name: str):
        assistant_responses = [response for d in batch 
                                for responses in d[f'ver_resp_{ver_model_name}_on_{gen_model_name}'] 
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
            
    def _get_dict_names(self, ver_model_name: str, gen_model_name: str, strict_ver_model_name: str):
        ver_resp = f'ver_resp_{ver_model_name}_on_{gen_model_name}'
        strict_ver_resp = f'strict_ver_resp_{strict_ver_model_name}_on_ver_{ver_model_name}_on_{gen_model_name}'
        fin_strict_ver_resp = f'fin_strict_ver_resp_{strict_ver_model_name}_on_ver_{ver_model_name}_on_{gen_model_name}'
        fin_ver_approval_bools = f'fin_ver_approval_bools_{strict_ver_model_name}_on_ver_{ver_model_name}_on_{gen_model_name}'
        fin_ver_approval_sums = f'fin_ver_approval_sums_{strict_ver_model_name}_on_ver_{ver_model_name}_on_{gen_model_name}'
        return ver_resp, strict_ver_resp, fin_strict_ver_resp, fin_ver_approval_bools, fin_ver_approval_sums

