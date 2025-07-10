from src.utils.model import ModelRunner
from tqdm import trange
from src.utils.dataset_manager import DatasetManager
from src.utils.util import extract_answer

class GenerationRunner(ModelRunner):
    def __init__(self, config: dict):
        super().__init__(config)
        self.templates = self.prompt_config['generator']['templates']
        self.batch_size = self.config['base_config']['batch_size']
        self.num_generations = self.config['base_config']['num_generations']

    def generate_solutions(self, datasetManager: DatasetManager):
        for model_name in self.gen_models:
            model_config = self.all_model_configs[model_name]
            chat_template = model_config['model']['chat_template']
            llm, sampling_params = self._load_model_and_sampling_params(model_config)
            assert sampling_params.n == self.num_generations, f"Number of generations {sampling_params.n} does not match {self.num_generations}"
            for i in trange (0, datasetManager.dataset_size, self.batch_size, desc=f"Generating {model_name} Task {datasetManager.task_id}"):
                # filepath = f"{self.output_base_path}/{self.dataset}/gen/task_{datasetManager.task_id}_solutions_{i}-{i+batch_size}.json"
                batch = datasetManager.dataset[i:i + self.batch_size]
                problems = self._get_problems(batch)
                # solutions = self._get_suggested_solutions(batch) # added 
                prompts = self._get_prompts(datasetManager.dataset_name, problems) # added prompts = self._get_prompts(datasetManager.dataset_name, problems, solutions) 
                user_prompts = [self._to_chat_prompt(chat_template, prompt) for prompt in prompts]
                outputs = self._generate(llm, sampling_params, user_prompts)
                self._post_processing(batch, outputs, model_config, datasetManager.dataset_name)
            llm = self._cleanup(llm)

    def _get_suggested_solutions(self, batch: list[dict]): # added
        is_batch = isinstance(batch, list)
        batch_solutions = [d["solution"] for d in batch] if is_batch else [batch["solution"]]
        return batch_solutions

    def _get_prompts(self, dataset_name: str, problem: str | list[str]):
        if dataset_name not in self.templates:
            raise ValueError(f"Unsupported dataset: '{dataset_name}'. Available datasets: {list(self.templates.keys())}")

        if isinstance(problem, list):
            return [self.templates[dataset_name]['prompt'].format(problem=p) for p in problem]
        return self.templates[dataset_name]['prompt'].format(problem=problem)
    
    def _post_processing(self, batch: list[dict], outputs: list[str], model_config: dict, dataset_name: str):
        model_name = model_config['model']['name']
        is_reasoning = model_config['model']['reasoning']
        for batch_idx, d in enumerate(batch):
            generated_solutions = [self._process_text(outputs[batch_idx].outputs[j].text, is_reasoning)
                                   for j in range(self.num_generations)]
            generated_answers = [extract_answer(solution, dataset_name) for solution in generated_solutions]
            d[f'generated_solutions_{model_name}'] = generated_solutions
            d[f'generated_answers_{model_name}'] = generated_answers

    # add methods here
