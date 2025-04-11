import argparse
import os
import yaml # Added for config loading
from openai import OpenAI, APIError # Added APIError
import time # Added for sleep
import itertools # Added for round-robin

# Function to load YAML config safely
def load_config(config_path):
    """Loads a YAML configuration file."""
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return None
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        return None
    except IOError as e:
        print(f"Error reading file {config_path}: {e}")
        return None

def get_server_endpoints(shared_dir: str, total_servers: int) -> dict[str, list[str]]:
    """Reads all server host:port endpoints from the common host files."""
    endpoints = {'generator': [], 'verifier': []}
    common_host_files = {
        'generator': os.path.join(shared_dir, "server_hosts.txt"),
        'verifier': os.path.join(shared_dir, "verifier_server_hosts.txt")
    }

    for key, file_path in common_host_files.items():
        print(f"[Client] Reading {key} server endpoints from {file_path}...")

        if not os.path.exists(file_path):
            print(f"[Client] Error: {key.capitalize()} host file {file_path} not found!")
            continue

        try:
            with open(file_path, "r") as f:
                # Read all lines, strip whitespace, filter empty lines
                endpoints[key] = [line.strip() for line in f if line.strip()]

            print(f"[Client] Found {len(endpoints[key])} {key} server endpoints.")
            print(f"[Client] {key.capitalize()} server endpoints: {endpoints[key]}")

        except IOError as e:
            print(f"[Client] Error reading {key} host file {file_path}: {e}")

    return endpoints

def run_distributed_inference(total_servers: int, config_file: str, shared_dir: str):
    """Runs inference against multiple vLLM servers in a distributed manner."""
    print(f"[Client] Starting distributed inference with {total_servers} servers.")

    # --- Load Main Configuration ---
    main_config = load_config(config_file)
    if main_config is None:
        print(f"[Client] Failed to load main config: {config_file}")
        return # Error already printed by load_config

    model_identifier = main_config.get('model')
    verifier_model_identifier = main_config.get('verifier_model')
    if not model_identifier:
        print(f"[Client] Error: 'model' identifier not found in main config {config_file}")
        return
    print(f"[Client] Using model identifier from main config: {model_identifier}")
    # --- End Load Main Configuration ---

    # --- Load Model Specific Configuration ---
    # Assuming script is run from repo root, construct relative path
    # Adjust if the execution directory is different
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # Adjust if structure changes
    model_config_path = os.path.join(repo_root, "src", "configs", "model", f"{model_identifier}.yaml")
    print(f"[Client] Attempting to load model config from: {model_config_path}")
    model_config = load_config(model_config_path)
    if verifier_model_identifier:
        verifier_model_config_path = os.path.join(repo_root, "src", "configs", "model", f"{verifier_model_identifier}.yaml")
        print(f"[Client] Attempting to load verifier model config from: {verifier_model_config_path}")
        verifier_model_config = load_config(verifier_model_config_path)
        verifier_model_path_in_config = verifier_model_config.get('model', {}).get('path')
        verifier_max_tokens = verifier_model_config.get('model', {}).get('max_new_tokens', 500)
        verifier_temperature = verifier_model_config.get('model', {}).get('temperature', 1.0)
    if model_config is None:
        print(f"[Client] Failed to load model config: {model_config_path}")
        return

    model_path_in_config = model_config.get('model', {}).get('path')
    if not model_path_in_config:
        print(f"[Client] Error: 'model.path' not found in model config {model_config_path}")
        return
    print(f"[Client] Using model path for API call: {model_path_in_config}")
    # --- End Load Model Specific Configuration ---

    # --- Discover and Initialize Server Clients ---
    server_endpoints = get_server_endpoints(shared_dir, total_servers)
    if not server_endpoints['generator']:
        print("[Client] Error: No generator server endpoints found. Exiting.")
        return
    if not server_endpoints['verifier']:
        print("[Client] Error: No verifier server endpoints found. Exiting.")
        return

    openai_api_key = "EMPTY" # vLLM doesn't require a key by default
    server_clients = {'generator': [], 'verifier': []}

    for key in server_clients:
        for server_host_port in server_endpoints[key]:
            openai_api_base = f"http://{server_host_port}/v1"
            try:
                client = OpenAI(
                    api_key=openai_api_key,
                    base_url=openai_api_base,
                    timeout=60.0, # Add a timeout
                )
                server_clients[key].append(client)
                print(f"[Client] Initialized OpenAI client for {server_host_port}")
            except Exception as e:
                print(f"[Client] Error initializing OpenAI client for {server_host_port}: {e}")
                return

    # --- End Discover and Initialize Server Clients ---

    # You could split your dataset here if needed, using task_id and num_tasks
    messages = [
        {"role": "user", "content": f"Explain quantum computing simply."}
    ]

    # Use round-robin to distribute requests across generator servers
    for task_id, client in enumerate(itertools.cycle(server_clients['generator'])):
        try:
            print(f"[Task {task_id}] Sending request to model '{model_path_in_config}' at {client.base_url}...")
            max_tokens = model_config.get('model', {}).get('max_new_tokens', 500)
            temperature = model_config.get('model', {}).get('temperature', 1.0)

            completion = client.chat.completions.create(
                model=model_path_in_config,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                n=1,
            )

            if completion.choices:
                result = completion.choices[0].message.content
                print(f"[Task {task_id}] Generator Response:\n---\n{result}\n---")

                # Pass the result to the verifier model
                verifier_message = [
                    {"role": "user", "content": f"Is this correct. Yes or No. Response: {result}"}
                ]
                verifier_client = server_clients['verifier'][task_id % len(server_clients['verifier'])]
                verifier_completion = verifier_client.chat.completions.create(
                    model=verifier_model_path_in_config,
                    messages=verifier_message,
                    max_tokens=verifier_max_tokens,
                    temperature=verifier_temperature,
                    n=1,
                )

                if verifier_completion.choices:
                    verifier_result = verifier_completion.choices[0].message.content
                    print(f"[Task {task_id}] Verifier Response:\n---\n{verifier_result}\n---")
                else:
                    print(f"[Task {task_id}] No choices returned in verifier completion.")
                return verifier_result

            else:
                print(f"[Task {task_id}] No choices returned in generator completion.")
                return None

        except APIError as e:
            print(f"[Task {task_id}] OpenAI API Connection Error connecting to {client.base_url}: {str(e)}")
            if e.response:
                try:
                    print(f"[Task {task_id}] Raw Response Body: {e.response.text}")
                except Exception:
                    print(f"[Task {task_id}] Could not read raw response body.")
            return None
        except Exception as e:
            print(f"[Task {task_id}] An unexpected error occurred during API call to {client.base_url}: {type(e).__name__}: {e}")
            return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-servers", type=int, required=True, help="Total number of server tasks.")
    parser.add_argument("--config-file", type=str, required=True, help="Path to the main configuration YAML file.")
    parser.add_argument("--shared-dir", type=str, required=True, help="Path to the shared directory for coordination files (e.g., host files).")
    args = parser.parse_args()

    # Basic validation
    if not os.path.isdir(args.shared_dir):
        print(f"Error: Shared directory not found or not a directory: {args.shared_dir}")
        exit(1)

    run_distributed_inference(
        total_servers=args.total_servers,
        config_file=args.config_file,
        shared_dir=args.shared_dir
    )
    print(f"[Client] Main script finished.")


