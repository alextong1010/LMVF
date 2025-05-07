import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trl.extras.vllm_client import VLLMClient
import re, itertools
from typing import List, Tuple
from transformers import AutoTokenizer
from src.utils.vera_utils import load_domain_specific_verifiers
from src.prompts.vera_prompts import get_vera_prompt, VERA_ASK_FOR_APPROVAL_ONLY_PROMPT
from datasets import Dataset
from termcolor import colored



# Ask the verifier to reply with the single word **True** or **False**
PROMPT_TEMPLATE = """You are a factual verifier.
If the model answer is entirely correct, reply with the single word **True**.
If it is wrong or partially wrong, reply with the single word **False**.

QUESTION:
{prompt}

MODEL ANSWER:
{completion}

Your reply:"""

TRUE_RE  = re.compile(r"\btrue\b",  re.I)
FALSE_RE = re.compile(r"\bfalse\b", re.I)

class LLMVerifier:
    """
    Send every (prompt,completion) to *every* vLLM server in `verifier_hosts_ports` and `strict_verifier_hosts_ports`.
    Reward = number of servers that replied "True" (0 … len(servers)).
    """
    def __init__(self,
                 verifier_hosts_ports: List[Tuple[str,int]],
                 verifier_model_name: str,
                 strict_verifier_hosts_ports: List[Tuple[str,int]],
                 strict_verifier_model_name: str,
                 config: dict,
                 verifier_model_config: dict,
                 strict_verifier_model_config: dict,
                 dataset: Dataset,
                 name: str = "llm_verifier"):
        self.verifier_clients = [VLLMClient(h, p, connection_timeout=30)
                        for h, p in verifier_hosts_ports]
        self.verifier_tokenizer = AutoTokenizer.from_pretrained(verifier_model_name)   # ← local tokenizer
        self.verifier_max_new_tokens = verifier_model_config['model']['max_new_tokens']
        self.strict_verifier_max_new_tokens = strict_verifier_model_config['model']['max_new_tokens']
        self.veras = load_domain_specific_verifiers(config['dataset'])
        self.direct_veras_idx = [i for i, v in enumerate(self.veras) if "direct" in v]
        self.config = config
        self.verifier_model_config = verifier_model_config
        self.strict_verifier_model_config = strict_verifier_model_config
        self.dataset = dataset
        self.__name__ = name
        self.curr_verifier_idx = 0
        self.curr_strict_verifier_idx = 0

        if strict_verifier_model_name != verifier_model_name:
            self.strict_verifier_clients = [VLLMClient(h, p, connection_timeout=30)
                            for h, p in strict_verifier_hosts_ports]
            self.strict_verifier_tokenizer = AutoTokenizer.from_pretrained(strict_verifier_model_name)
        else:
            self.strict_verifier_clients = self.verifier_clients
            self.strict_verifier_tokenizer = self.verifier_tokenizer

    # ------------------------------------------------------------------
    def _batch_generate(self, client, texts: List[str], strict: bool = False):
        """helper – returns list[str] decoded generations"""
        ids = client.generate(
            prompts=texts,
            n=1,
            max_tokens=self.verifier_max_new_tokens if not strict else self.strict_verifier_max_new_tokenss,
            temperature=self.verifier_model_config['model']['temperature'] if not strict else self.strict_verifier_model_config['model']['temperature'],
        )
        if not strict:
            outputs = [self.verifier_tokenizer.decode(seq, skip_special_tokens=True) for seq in ids]
        else:
            outputs = [self.strict_verifier_tokenizer.decode(seq, skip_special_tokens=True) for seq in ids]
        return outputs


    # ------------------------------------------------------------------
    def __call__(self,
                 prompts:     List[str],
                 completions: List[str],
                 **kwargs) -> List[float]:
        """Return len(prompts) rewards (float, but actually int 0…N)."""

        use_verifier_idx = self.curr_verifier_idx%len(self.verifier_clients)
        use_strict_verifier_idx = self.curr_strict_verifier_idx%len(self.strict_verifier_clients)

        # Extract the full text first
        raw_texts = [p[0]['content'][0]['text'] for p in prompts]
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

        answers = [c[0]['content'] for c in completions]
        batch_prompts = []
        for question, answer in zip(questions, answers):
            for vera_name in self.veras:
                batch_prompts.append(get_vera_prompt(self.config['dataset'], vera_name, question, answer))
        
        assert len(batch_prompts) == len(prompts) * len(self.veras), colored(f"Number of prompts is not equal to the number of veras * batch size. Double check!", "red") 

        inputs = [
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                },
            ] for prompt in batch_prompts
        ]
        verifier_outputs = self._batch_generate(self.verifier_clients[use_verifier_idx], inputs, strict=False)

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
        
        if not strict_inputs:
            print(colored("Warning: No strict prompts generated (maybe all verifiers were direct?). Proceeding to finalize outputs.", "yellow"))
            strict_decoded_outputs = []
        else:
            print(f"\nGetting {len(strict_inputs)} strict True/False approvals for non-direct verifier responses in the batch...\n")
            strict_verifier_outputs = self._batch_generate(self.strict_verifier_clients[use_strict_verifier_idx], strict_inputs, strict=True)

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
                    
        # Extract the True/False approvals from the strict verifier outputs


        self.curr_verifier_idx += 1
        self.curr_strict_verifier_idx += 1

        # convert to float because GRPO expects list[float]
        return [float(v) for v in votes_true]
