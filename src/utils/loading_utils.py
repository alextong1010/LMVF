import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset_utils import get_dataset_config
from vllm.utils import FlexibleArgumentParser
from vllm.engine.arg_utils import EngineArgs
import yaml

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_domain_specific_verifiers(dataset_name):
    config = get_dataset_config()
    if dataset_name in config and "verifiers" in config[dataset_name]:
        return config[dataset_name]["verifiers"]
    else:
        raise ValueError(f"Domain-specific verifiers not implemented for dataset {dataset_name}.")

def parse_initial_arguments(repo_root):
    parser = FlexibleArgumentParser()
    parser.add_argument(
        "--config-path",
        default="configs/eval_config.yaml",
        type=str,
        help="Config Path",
    )
    parser.add_argument(
        "--task-id",
        default=0,
        type=int,
        help="Task ID (defaults to 0 for single gpu tasks)",
    )
    parser.add_argument(
        "--num-tasks",
        default=1,
        type=int,
        help="Number of tasks",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        type=bool,
        help="Verbose output",
    )
    config_args, remaining_vllm_args = parser.parse_known_args()
    config_args.config_path = os.path.join(repo_root, config_args.config_path)
    return config_args, remaining_vllm_args

def load_and_validate_config(config_args, repo_root):
    config = load_config(config_args.config_path)

    model_name = config.get("model")
    model_config_path = os.path.join(repo_root, "src", "configs", "model", f"{model_name}.yaml")
    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"Model config file not found: {model_config_path}")
    model_config = load_config(model_config_path)
    model_path = model_config.get("model", {}).get("path")
    if not model_path:
        raise ValueError(f"Model path not found in {model_config_path}")
    
    verifier_model_name = config.get("verifier_model")
    if verifier_model_name:
        verifier_model_config_path = os.path.join(repo_root, "src", "configs", "model", f"{verifier_model_name}.yaml")
        if not os.path.exists(verifier_model_config_path):
            raise FileNotFoundError(f"Verifier model config file not found: {verifier_model_config_path}")
        verifier_model_config = load_config(verifier_model_config_path)
        verifier_model_path = verifier_model_config.get("model", {}).get("path")
        if not verifier_model_path:
            raise ValueError(f"Verifier model path not found in {verifier_model_config_path}")
    else:
        verifier_model_config = None
        verifier_model_path = None

    # Add strict verifier model loading
    strict_verifier_model_name = config.get("strict_verifier_model")
    if strict_verifier_model_name:
        strict_verifier_model_config_path = os.path.join(repo_root, "src", "configs", "model", f"{strict_verifier_model_name}.yaml")
        if not os.path.exists(strict_verifier_model_config_path):
            raise FileNotFoundError(f"Strict verifier model config file not found: {strict_verifier_model_config_path}")
        strict_verifier_model_config = load_config(strict_verifier_model_config_path)
        strict_verifier_model_path = strict_verifier_model_config.get("model", {}).get("path")
        if not strict_verifier_model_path:
            raise ValueError(f"Strict verifier model path not found in {strict_verifier_model_config_path}")
    else:
        strict_verifier_model_config = None
        strict_verifier_model_path = None

    return config, model_config, model_path, verifier_model_config, verifier_model_path, strict_verifier_model_config, strict_verifier_model_path

def parse_vllm_args(model_path, remaining_args=None):
    parser = FlexibleArgumentParser()
    engine_group = parser.add_argument_group("Engine arguments")
    EngineArgs.add_cli_args(engine_group)
    engine_group.set_defaults(model=model_path)  # Set default model from config
    args = vars(parser.parse_args(remaining_args))
    return args