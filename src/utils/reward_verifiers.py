import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List
from src.utils.vera_utils import load_domain_specific_verifiers
from src.prompts.vera_prompts import get_vera_prompt, VERA_ASK_FOR_APPROVAL_ONLY_PROMPT
from datasets import Dataset
from termcolor import colored
from openai import OpenAI

class LLMVerifier:
    """
    Send every (prompt,completion) to *every* vLLM server in `verifier_hosts_ports` and `strict_verifier_hosts_ports`. 
    Currently, only one server is used because having multiple servers is unnecessary.
    Reward = number of servers that replied "True" (0 … len(servers)).
    """
    def __init__(self,
                 verifier_hosts_port: str,
                 verifier_model_path: str,
                 strict_verifier_hosts_port: str,
                 strict_verifier_model_path: str,
                 config: dict,
                 verifier_model_config: dict,
                 strict_verifier_model_config: dict,
                 dataset: Dataset,
                 prompt_version: int,
                 name: str = "llm_verifier"):
        
        openai_api_key = "EMPTY" # vLLM doesn't require a key by default
        verifier_openai_api_base = f"http://{verifier_hosts_port}/v1"
        strict_openai_api_base = f"http://{strict_verifier_hosts_port}/v1"


        self.verifier_client = OpenAI(
                    api_key=openai_api_key,
                    base_url=verifier_openai_api_base,
                    timeout=60.0, # Add a timeout
                )
        self.verifier_max_new_tokens = verifier_model_config['model']['max_new_tokens']
        self.strict_verifier_max_new_tokens = 2048 if strict_verifier_model_config['model']['reasoning'] else 16
        self.veras = load_domain_specific_verifiers(config['dataset'])
        self.direct_veras_idx = [i for i, v in enumerate(self.veras) if "direct" in v]
        self.config = config
        self.verifier_model_config = verifier_model_config
        self.strict_verifier_model_config = strict_verifier_model_config
        self.dataset = dataset
        self.__name__ = name
        self.curr_verifier_idx = 0
        self.curr_strict_verifier_idx = 0
        self.prompt_version = prompt_version
        
        if strict_verifier_model_path != verifier_model_path:
            self.strict_verifier_client = OpenAI(
                    api_key=openai_api_key,
                    base_url=strict_openai_api_base,
                    timeout=60.0, # Add a timeout
                )
        else:
            self.strict_verifier_client = self.verifier_client

    # ------------------------------------------------------------------
    def _batch_generate(self, client, texts: List[str], strict: bool = False):
        """helper – returns list[str] decoded generations"""
        breakpoint()
        outputs = []
        for text in texts:
            outputs.append(client.chat.completions.create(
                model=self.verifier_model_config['model']['path'] if not strict else self.strict_verifier_model_config['model']['path'],
                messages=text,
                max_tokens=self.verifier_max_new_tokens if not strict else self.strict_verifier_max_new_tokens,
                temperature=self.verifier_model_config['model']['temperature'] if not strict else self.strict_verifier_model_config['model']['temperature'],
                n=1,
            ))
        breakpoint()
        return outputs


    # ------------------------------------------------------------------
    def __call__(self,
                 prompts:     List[str],
                 completions: List[str],
                 **kwargs) -> List[float]:
        """Return len(prompts) rewards (float, but actually int 0…N)."""

        # Extract the full text first, could be any of the following formats.
        if self.prompt_version == 1:
            raw_texts = [p[0]['content'][0]['text'] for p in prompts]
        elif self.prompt_version == 2:
            raw_texts = [p[0]['content'] for p in prompts]
        elif self.prompt_version == 3:
            raw_texts = prompts
        else:
            raise ValueError(f"Unsupported prompt version: {self.prompt_version}")

        questions = []
        for text in raw_texts:
            try:
                # Find the text between 'QUESTION:' and 'Provide your detailed solution below:'
                after_question_marker = text.split('QUESTION:')[1]
                question_text = after_question_marker.split('\n\nProvide your detailed solution below:')[0]
                questions.append(question_text.strip())
            except IndexError:
                # Fallback or logging if parsing fails
                print(f"Warning: Could not parse question accurately from text: {text}")
                questions.append(text) # Using full text as fallback for now
        
        if self.prompt_version == 1:
            answers = [c[0]['content'][0]['text'] for c in completions]
        elif self.prompt_version == 2:
            answers = [c[0]['content'] for c in completions]
        elif self.prompt_version == 3:
            answers = completions
        
        batch_prompts = []
        for question, answer in zip(questions, answers):
            for vera_name in self.veras:
                batch_prompts.append(get_vera_prompt(self.config['dataset'], vera_name, question, answer))
        
        assert len(batch_prompts) == len(prompts) * len(self.veras), colored(f"Number of prompts is not equal to the number of veras * batch size. Double check!", "red") 

        inputs = [
            [
                {
                    "role": "user",
                    "content": prompt
                },
            ] for prompt in batch_prompts
        ]
        verifier_outputs = self._batch_generate(self.verifier_client, inputs, strict=False)
        breakpoint()
        strict_inputs = []
        original_indices = []
        global_response_idx = 0
        for vera_idx, verifier_output in enumerate(verifier_outputs):
            if vera_idx%len(self.veras) not in self.direct_veras_idx:
                vera_name = self.veras[vera_idx%len(self.veras)]
                strict_prompt = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": batch_prompts[vera_idx]}]
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": verifier_output}]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": VERA_ASK_FOR_APPROVAL_ONLY_PROMPT}]
                    }
                ]
                strict_inputs.append(strict_prompt)
                original_indices.append(global_response_idx)
            global_response_idx += 1
        breakpoint()
        
        if not strict_inputs:
            print(colored("Warning: No strict prompts generated (maybe all verifiers were direct?). Proceeding to finalize outputs.", "yellow"))
            strict_decoded_outputs = []
        else:
            print(f"\nGetting {len(strict_inputs)} strict True/False approvals for non-direct verifier responses in the batch...\n")
            strict_verifier_outputs = self._batch_generate(self.strict_verifier_client, strict_inputs, strict=True)

            strict_uses_reasoning = self.strict_verifier_model_config and self.strict_verifier_model_config.get("model", {}).get("reasoning")

            strict_decoded_outputs = []
            for output in strict_verifier_outputs:
                response_text = output.outputs[0].text or ""
                response_text = response_text.split("<end_of_turn>")[0].split("<|eot_id|>")[0].strip()

                if strict_uses_reasoning:
                    try:
                        response = response_text.split("</think>")[1]
                    except IndexError:
                        response = response_text
                    strict_decoded_outputs.append(response)
                else:
                    strict_decoded_outputs.append(response_text)
        breakpoint()
        # Combine initial and strict responses to get final outputs
        final_outputs_flat = {}
        strict_output_idx = 0
        current_global_idx = 0
        for sol_idx, solution_responses in enumerate(verifier_outputs):
            for vera_idx, initial_response in enumerate(solution_responses):
                if vera_idx in self.direct_veras_idx:
                    final_outputs_flat[current_global_idx] = initial_response
                else:
                    try:
                        original_idx_pos = original_indices.index(current_global_idx)
                        if original_idx_pos < len(strict_decoded_outputs):
                            final_outputs_flat[current_global_idx] = strict_decoded_outputs[original_idx_pos]
                        else:
                            print(colored(f"Error: Index mismatch finding strict output for global index {current_global_idx}", "red"))
                            final_outputs_flat[current_global_idx] = "# STRICT OUTPUT ERROR #"
                    except ValueError:
                        # This index was not in original_indices, should not happen if logic is correct
                        print(colored(f"Error: Global index {current_global_idx} not found in original_indices for strict mapping.", "red"))
                        final_outputs_flat[current_global_idx] = "# STRICT INDEX ERROR #"

                current_global_idx += 1

        # Reshape final_outputs_flat back into the batch structure
        current_read_idx = 0
        breakpoint()
        # Extract the True/False approvals from the strict verifier outputs


        self.curr_verifier_idx += 1
        self.curr_strict_verifier_idx += 1

        # convert to float because GRPO expects list[float]
        return None
