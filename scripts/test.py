# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser
VERA_ANSWER_SYMBOL = "FINAL VERIFICATION ANSWER:"



def main(args: dict):
    # Pop arguments not used by LLM
    max_tokens = args.pop("max_tokens")
    # args["enable_reasoning"] = True
    # args["reasoning_parser"] = "deepseek_r1"
    # Create an LLM
    llm = LLM(**args)

    # Create sampling params object
    sampling_params = llm.get_default_sampling_params()
    breakpoint()
    if max_tokens is not None:
        sampling_params.max_tokens = max_tokens

    def print_outputs(outputs):
        print("\nGenerated Outputs:\n" + "-" * 80)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}\n")
            print(f"Generated text: {generated_text!r}")
            print("-" * 80)

    print("=" * 80)

    # In this script, we demonstrate how to pass input to the chat method:
    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": "Hello."
        },
        {
            "role": "assistant",
            "content": "The solution is correct. **Final Answer:** $\\boxed{42}$ **Verification:** 1. The regular hexagon is divided into six equilateral triangles, each with a perimeter of 21 inches. 2. Each triangle's perimeter is 3s = 21, leading to s = 7 inches. 3. The hexagon's perimeter is 6s = 6*7 = 42 inches. 4. The reasoning aligns with the properties of a regular hexagon and equilateral triangles. Thus, the final answer is $\\boxed{42}$, which is correct.",
        },
        {
            "role": "user",
            "content": f"To clarify, based on the above analysis, reply with ONLY '{VERA_ANSWER_SYMBOL}True' or ONLY '{VERA_ANSWER_SYMBOL}False'. Do not include any other text in your response."
            ,
        },
    ]

    # You can run batch inference with llm.chat API
    conversations = [conversation for _ in range(10)]

    # We turn on tqdm progress bar to verify it's indeed running batch inference
    outputs = llm.chat(conversations, sampling_params, use_tqdm=True)
    print_outputs(outputs)

    # A chat template can be optionally supplied.
    # If not, the model will use its default chat template.

    breakpoint()

if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    # Add engine args
    engine_group = parser.add_argument_group("Engine arguments")
    EngineArgs.add_cli_args(engine_group)
    engine_group.set_defaults(model="google/gemma-3-1b-it")
    # engine_group.set_defaults(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

    # Add sampling params
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", default=1024, type=int)
    # Add example params
    args: dict = vars(parser.parse_args())
    breakpoint()
    main(args)