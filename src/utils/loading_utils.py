import torch
from utils.dataset_utils import get_dataset_config
from openai import OpenAI
import os
import yaml
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_models(config):
    """
    Load the generator and verifier models based on the configuration.
    
    Args:
        config (dict): Configuration dictionary containing model information
        
    Returns:
        tuple: (generator_model, verifier_model)
    """
    print("Loading models and tokenizers...")
    
    # Load generator model
    generator_model, max_tokens, temperature, num_generations, model_path_in_config = load_model(config)
    
    # Load verifier model if specified, otherwise use the generator model
    if config.get('verifier_model'):
        print(f"Using separate verifier model: {config['verifier_model']}")
        verifier_model, max_tokens, temperature, num_generations, model_path_in_config = load_model(config)
    else:
        print("Using the same model for generation and verification")
        verifier_model = generator_model
    
    return generator_model, verifier_model, max_tokens, temperature, num_generations, model_path_in_config

def load_model(config):
    """
    Load a single model.
    
    Args:
        config (dict): Configuration dictionary containing model information

    Returns:
        model: The loaded model
    """
    print(f"Downloading model: {config['model']}...")
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # Adjust if structure changes
    model_config_path = os.path.join(repo_root, "configs", "model", f"{config['model']}.yaml")
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
    model_path_in_config = model_config.get('model', {}).get('path')
    if not model_path_in_config:
        print(f"[Task {config['task_id']}] Error: 'model.path' not found in model config {model_config_path}")
        return
    print(f"[Task {config['task_id']}] Using model path for API call: {model_path_in_config}")

    # Construct path to task-specific host file using the provided shared directory
    host_file = os.path.join(config['shared_dir'], f"server_host_{config['task_id']}.txt")
    print(f"[Task {config['task_id']}] Reading host file: {host_file}")

    # Load server host:port from task-specific file
    try:
        with open(host_file, "r") as f:
            server_host_port = f.read().strip()
        if not server_host_port or ':' not in server_host_port:
             print(f"[Task {config['task_id']}] Error: Invalid content in host file {host_file}: '{server_host_port}'")
             return
    except IOError as e:
        print(f"[Task {config['task_id']}] Error reading host file {host_file}: {e}")
        return
    
    print(f"[Task {config['task_id']}] Connecting to vLLM server at {server_host_port}")

    openai_api_key = "EMPTY" # vLLM doesn't require a key by default
    # Construct base URL correctly
    openai_api_base = f"http://{server_host_port}/v1"

    # Initialize OpenAI-compatible client
    try:
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=60.0, # Add a timeout
        )
    except Exception as e:
        print(f"[Task {config['task_id']}] Error initializing OpenAI client: {e}")
        return
    
    return client

def load_domain_specific_verifiers(dataset_name):
    config = get_dataset_config()
    if dataset_name in config and "verifiers" in config[dataset_name]:
        return config[dataset_name]["verifiers"]
    else:
        raise ValueError(f"Domain-specific verifiers not implemented for dataset {dataset_name}.")