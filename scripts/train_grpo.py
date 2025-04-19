import argparse

from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

def main():
    print("Starting train_grpo.py")
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm_server_host", type=str, default="", help="The server IP")
    args = parser.parse_args()

    # Example dataset from TLDR
    dataset = load_dataset("trl-lib/tldr", split="train")

    # Dummy reward function: count the number of unique characters in the completions
    def reward_num_unique_chars(completions, **kwargs):
        return [len(set(c)) for c in completions]

    training_args = GRPOConfig(
        output_dir="gemma-3-1b-it-GRPO",
        per_device_train_batch_size=4,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        use_vllm=True,
        vllm_server_host=args.vllm_server_host.replace("ip-", "").replace("-", "."),  # from ip-X-X-X-X to X.X.X.X
    )

    trainer = GRPOTrainer(model="google/gemma-3-1b-it", args=training_args, reward_funcs=reward_num_unique_chars, train_dataset=dataset)
    trainer.train()

if __name__=="__main__":
    main()