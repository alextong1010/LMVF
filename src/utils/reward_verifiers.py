from trl.extras.vllm_client import VLLMClient
import re, itertools
from typing import List, Tuple

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
    Send every (prompt,completion) to *every* vLLM server in `hosts_ports`.
    Reward = number of servers that replied "True" (0 … len(servers)).
    """
    def __init__(self,
                 hosts_ports: List[Tuple[str,int]],
                 max_new_tokens: int = 2):
        self.clients = [VLLMClient(h, p, connection_timeout=30)
                        for h, p in hosts_ports]
        self.max_new_tokens = max_new_tokens

    # ------------------------------------------------------------------
    def _batch_generate(self, client, texts: List[str]):
        """helper – returns list[str] decoded generations"""
        ids = client.generate(
            prompts=texts,
            n=1,
            max_tokens=self.max_new_tokens,
            temperature=0.0,    # deterministic
        )
        return [client.tokenizer.decode(x, skip_special_tokens=True) for x in ids]

    # ------------------------------------------------------------------
    def __call__(self,
                 prompts:     List[str],
                 completions: List[str],
                 **kwargs) -> List[float]:
        """Return len(prompts) rewards (float, but actually int 0…N)."""
        batch_texts = [PROMPT_TEMPLATE.format(prompt=p, completion=c)
                       for p, c in zip(prompts, completions)]

        # Collect votes from every verifier --------------------------------
        votes_true = [0] * len(prompts)          # will accumulate 1 per 'True'

        for client in self.clients:
            replies = self._batch_generate(client, batch_texts)

            for i, reply in enumerate(replies):
                if TRUE_RE.search(reply):
                    votes_true[i] += 1
                elif FALSE_RE.search(reply):
                    pass                         # explicit False → +0
                else:
                    # un‑parsable → treat as False, could log if you wish
                    pass

        # convert to float because GRPO expects list[float]
        return [float(v) for v in votes_true]
