import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List
from src.utils.vera_utils import load_domain_specific_verifiers
from src.prompts.vera_prompts import get_vera_prompt, VERA_ASK_FOR_APPROVAL_ONLY_PROMPT
from datasets import Dataset
from termcolor import colored
from transformers import AutoTokenizer
from trl.extras.vllm_client import VLLMClient
from trl.data_utils import maybe_apply_chat_template
import json
import numpy as np
from src.utils.vera_utils import extract_verifier_approval
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
                 logging: bool = False,
                 output_dirpath: str = "",
                 batch_size: int = 1,
                 name: str = "llm_verifier"):
        verifier_hosts_port = verifier_hosts_port.split(":")
        self.verifier_client = VLLMClient(verifier_hosts_port[0], int(verifier_hosts_port[1]), connection_timeout=30)
        self.verifier_tokenizer = AutoTokenizer.from_pretrained(verifier_model_path)
        self.verifier_max_new_tokens = verifier_model_config['model']['max_new_tokens']
        self.strict_verifier_max_new_tokens = 2048 if strict_verifier_model_config['model']['reasoning'] else 16
        self.veras = load_domain_specific_verifiers(config['dataset'])
        self.direct_veras_idx = [i for i, v in enumerate(self.veras) if "direct" in v]
        self.config = config
        self.verifier_model_config = verifier_model_config
        self.strict_verifier_model_config = strict_verifier_model_config
        self.dataset = list(dataset)
        self.__name__ = name
        self.curr_verifier_idx = 0
        self.curr_strict_verifier_idx = 0
        self.batch_size = batch_size
        self.prompt_version = prompt_version
        self.logging = logging
        self.output_dirpath = output_dirpath
        self.num_veras = len(self.veras)
        self.num_prompts_per_batch = self.config['per_device_train_batch_size'] // self.config['num_generations']
        self.num_generations = self.config['num_generations']
        self.num_veras_per_batch = self.num_prompts_per_batch * self.num_generations * self.num_veras
        self.rank_bonus = [0.025 * (1 - i / (self.config['per_device_train_batch_size'] - 1)) if self.config['per_device_train_batch_size'] > 1 else 0.025 for i in range(self.config['per_device_train_batch_size'])]

        strict_verifier_hosts_port = strict_verifier_hosts_port.split(":")
        if strict_verifier_model_path != verifier_model_path:
            self.strict_verifier_client = VLLMClient(strict_verifier_hosts_port[0], int(strict_verifier_hosts_port[1]), connection_timeout=30)
            self.strict_verifier_tokenizer = AutoTokenizer.from_pretrained(strict_verifier_model_path)
        else:
            self.strict_verifier_client = self.verifier_client
            self.strict_verifier_tokenizer = self.verifier_tokenizer
    # ------------------------------------------------------------------
    def _batch_generate(self, client, texts: List[str], strict: bool = False):
        """helper – returns list[str] decoded generations"""
        prompts_text = [maybe_apply_chat_template(example, self.verifier_tokenizer)['prompt'] for example in texts]

        ids = client.generate(
            prompts=prompts_text,
            n=1,
            max_tokens=self.verifier_max_new_tokens if not strict else self.strict_verifier_max_new_tokens,
            temperature=self.verifier_model_config['model']['temperature'] if not strict else self.strict_verifier_model_config['model']['temperature'],
        )
        if not strict:
            outputs = self.verifier_tokenizer.batch_decode(ids, skip_special_tokens=True)
        else:
            outputs = self.strict_verifier_tokenizer.batch_decode(ids, skip_special_tokens=True)
        return outputs


    # ------------------------------------------------------------------
    def __call__(self,
                 prompts:     List[str],
                 completions: List[str],
                 problem: List[str],
                 level: List[str],
                 type: List[str],
                 solution: List[str],
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

        assert len(batch_prompts) == self.num_veras_per_batch, colored(f"Number of prompts is not equal to the number of veras * batch size. Double check!", "red") 

        if self.prompt_version == 1:
            inputs = [
                {
                    "prompt": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}]
                        },
                    ]
                }
                for prompt in batch_prompts
            ]
        elif self.prompt_version == 2:
            inputs = [
                {
                    "prompt": [
                        {
                            "role": "user",
                            "content": prompt
                        },
                    ]
                }
                for prompt in batch_prompts
            ]

        verifier_outputs = self._batch_generate(self.verifier_client, inputs, strict=False)
        verifier_uses_reasoning = self.verifier_model_config and self.verifier_model_config.get("model", {}).get("reasoning")
        if verifier_uses_reasoning:
            verifier_decoded_outputs = [
                    output.split("\n</think>\n")[1] if "\n</think>\n" in output else output
                    for output in verifier_outputs
                ]
        else:
            verifier_decoded_outputs = verifier_outputs[:]

        # Generate strict inputs and original indices
        strict_inputs = []
        original_indices = []
        global_response_idx = 0
        for vera_idx, verifier_decoded_output in enumerate(verifier_decoded_outputs):
            if vera_idx%len(self.veras) not in self.direct_veras_idx:
                vera_name = self.veras[vera_idx%len(self.veras)]
                if self.prompt_version == 1:
                    strict_prompt = { 'prompt': 
                            [
                                {
                                    "role": "user",
                                    "content": [{"type": "text", "text": batch_prompts[vera_idx]}]
                                },
                                {   
                                    "role": "assistant",
                                    "content": [{"type": "text", "text": verifier_decoded_output}]
                                },
                                {
                                    "role": "user",
                                    "content": [{"type": "text", "text": VERA_ASK_FOR_APPROVAL_ONLY_PROMPT}]
                                }
                            ]
                        }
                    
                elif self.prompt_version == 2:
                    strict_prompt = {
                            "prompt": [
                                {
                                    "role": "user",
                                    "content": batch_prompts[vera_idx]
                                },
                                {
                                    "role": "assistant",
                                    "content": verifier_decoded_output
                                },  
                                {
                                    "role": "user",
                                    "content": VERA_ASK_FOR_APPROVAL_ONLY_PROMPT
                                }
                            ]
                        }   
                    
                strict_inputs.append(strict_prompt)
                original_indices.append(global_response_idx)
            global_response_idx += 1
        
        if not strict_inputs:
            print(colored("Warning: No strict prompts generated (maybe all verifiers were direct?). Proceeding to finalize outputs.", "yellow"))
            strict_decoded_outputs = []
        else:
            print(f"\nGetting {len(strict_inputs)} strict True/False approvals for non-direct verifier responses in the batch...\n")
            strict_verifier_outputs = self._batch_generate(self.strict_verifier_client, strict_inputs, strict=True)
            strict_uses_reasoning = self.strict_verifier_model_config and self.strict_verifier_model_config.get("model", {}).get("reasoning")

            if strict_uses_reasoning:
                strict_decoded_outputs = [
                    output.split("\n</think>\n")[1] if "\n</think>\n" in output else output
                    for output in strict_verifier_outputs
                ]
            else:
                strict_decoded_outputs = strict_verifier_outputs[:]

        assert len(strict_decoded_outputs) == len(inputs), colored(f"Number of strict decoded outputs is not equal to the number of original inputs. Double check!", "red")

        # Apply replacements where vera_idx matches
        final_outputs = []
        for i in range(len(strict_decoded_outputs)):
            vera_idx = i % self.num_veras
            if vera_idx in self.direct_veras_idx:
                final_outputs.append(verifier_decoded_outputs[i])
            else:
                final_outputs.append(strict_decoded_outputs[i])

        assert len(final_outputs) == self.num_veras_per_batch, colored(f"Number of final outputs is not equal to the number of veras * batch size. Double check!", "red")

        # Chunk into vera units (each of length 6)
        reshaped_final_outputs = [final_outputs[i:i + self.num_veras] for i in range(0, len(final_outputs), self.num_veras)]

        verifier_approval_bools = [[extract_verifier_approval(output) for output in g] for g in reshaped_final_outputs]

        summed_verifier_approval_bools = [sum(inner) for inner in verifier_approval_bools]


        scaled_reward = [(r / 6) ** 2 for r in summed_verifier_approval_bools]

        result_shaped = [b + r for b, r in zip(scaled_reward, self.rank_bonus)]

        grouped_strict_decoded_outputs = [strict_decoded_outputs[i:i + self.num_veras] for i in range(0, len(strict_decoded_outputs), self.num_veras)]
        grouped_verifier_decoded_outputs = [verifier_decoded_outputs[i:i + self.num_veras] for i in range(0, len(verifier_decoded_outputs), self.num_veras)]
        breakpoint()
        log_data_list = [
            {
                "problem": p,
                "level": lev,
                "type": typ,
                "solution": sol,
                "answers": ans,
                "verifier_decoded_output": ver_out,
                "strict_decoded_output": strict_out,
                "summed_verifier_approval_bools": ver_approval,
                "scaled_reward": scaled,
                "result_shaped": res_shaped
            }
            for p, lev, typ, sol, ans, ver_out, strict_out, ver_approval, scaled, res_shaped in zip(
                problem, level, type, solution, answers, grouped_verifier_decoded_outputs, grouped_strict_decoded_outputs, summed_verifier_approval_bools, scaled_reward, result_shaped
            )
        ]
        
        if self.logging:
            print(f"Logging {len(log_data_list)} log data to {self.output_dirpath}")
            with open(f"{self.output_dirpath}/task_{self.curr_verifier_idx}-{self.curr_verifier_idx+self.batch_size}.json", "w") as f:
                json.dump(log_data_list, f)
            
            print("Result Shaped: ", result_shaped)

        self.curr_verifier_idx += self.batch_size
        self.curr_strict_verifier_idx += self.batch_size

        # convert to float because GRPO expects list[float]
        return result_shaped

