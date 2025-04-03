from openai import OpenAI

def run_inference():
    # Load server host from shared file
    with open("shared_dir/gemma_test/server_host.txt", "r") as f:
        server_host = f.read().strip()

    print(f"Connecting to vLLM server at {server_host}:8000")

    openai_api_key = "EMPTY"
    openai_api_base = f"http://{server_host}:8000/v1"
    # Initialize OpenAI-compatible client
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # Chat-style prompt
    messages = [
        {"role": "user", "content": "Explain quantum computing in simple terms:"}
    ]

    # Call /v1/chat/completions endpoint
    completion = client.chat.completions.create(
        model="google/gemma-3-1b-it",  # Change if your model has a different ID
        messages=messages,
        max_tokens=500,
        temperature=1.0,
        n=2,
    )

    # Print the output
    print(f"Generated response: {completion.choices[0].message.content}")
    return completion.choices[0].message.content

if __name__ == "__main__":
    run_inference()

