from abc import ABC
from vllm import LLM
from vllm.distributed.parallel_state import destroy_model_parallel
import torch.distributed as dist
import gc
import torch
import os
import yaml
import json

class ModelRunner(ABC):
    """
    Base class for generators, verifiers, and recombinators.
    """
    def __init__(self, config: dict):
        self.config = config
        self.gen_models = self.config['base_config']['models']
        if isinstance(self.gen_models, str):
            self.gen_models = [self.gen_models]
        self._load_prompt_config()
        self._load_all_model_configs()
        self.output_base_path = self.config['base_config']['output_base_path']
        self.dataset = self.config['base_config']['dataset']

    def _to_chat_prompt(self, chat_template: str, prompt: str, assistant: bool = False):
        # Image currently not supported
        if chat_template == "text_only":
            if assistant:
                return [{"role": "assistant", "content": prompt}]
            else:
                return [{"role": "user", "content": prompt}]
        elif chat_template == "text_and_image":
            if assistant:
                return [{"role": "assistant", "content": [{"type": "text", "text": prompt}]}]
            else:
                return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        elif chat_template == "None" or chat_template == None:
            return prompt
        else:
            raise NotImplementedError(f"Unsupported chat template: {chat_template}")

    def _get_problems(self, batch: list[dict] | dict):
        is_batch = isinstance(batch, list)
        batch_problems = [d["problem"] for d in batch] if is_batch else [batch["problem"]]
        return batch_problems

    def _load_all_model_configs(self):
        model_configs = {}
        model_config_dir = os.path.join(os.path.dirname(__file__), '..', 'configs', 'model')

        for filename in os.listdir(model_config_dir):
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                model_name = os.path.splitext(filename)[0]
                model_config_path = os.path.join(model_config_dir, filename)
                with open(model_config_path, 'r') as f:
                    model_config = yaml.safe_load(f)
                model_configs[model_name] = model_config

        self.all_model_configs = model_configs

    def _load_prompt_config(self):
        prompts_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'prompts.yaml')
        with open(prompts_path, 'r') as f:
            self.prompt_config = yaml.safe_load(f)

    def _load_model_and_sampling_params(self, model_config: dict):
        # load model configs and model args
        model_path = model_config["model"]["path"]
        enable_reasoning = model_config["model"].get("reasoning", False)
        reasoning_parser = model_config["model"].get("reasoning_parser", None)
        llm = LLM(model=model_path, enable_reasoning=enable_reasoning, reasoning_parser=reasoning_parser)
        sampling_params = llm.get_default_sampling_params()
        sampling_params.max_tokens = model_config.get("model", {}).get("max_new_tokens", 2048)
        sampling_params.n = self.config.get("base_config", {}).get("num_generations", 1)
        return llm, sampling_params

    def _generate(self, llm: LLM, sampling_params: dict, inputs: list[list[dict]] | list[str]):
        outputs = llm.chat(inputs, sampling_params, use_tqdm=False)
        return outputs
    
    def _process_text(self, text: str, is_reasoning: bool):
        if is_reasoning and "</think>" in text:
            return text.split("</think>")[1].split("<end_of_turn>")[0]
        else:
            return text.split("<end_of_turn>")[0]
    
    def _save_outputs(self, outputs: list[dict], filepath: str):
        with open(filepath, "w") as f:
            json.dump(outputs, f)

    def _cleanup(self, model: LLM):
        if model is None:
            return None
        
        # Let vLLM dismantle model‑parallel state
        destroy_model_parallel()
        
        # Delete the reference
        del model

        # Run the CPython GC & clear PyTorch's allocator
        gc.collect()
        torch.cuda.empty_cache()

        # Close the NCCL pg if it exists
        if dist.is_initialized():
            dist.destroy_process_group()

        return None

