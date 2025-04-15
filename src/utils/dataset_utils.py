from datasets import load_dataset
import yaml
import os
import json
from utils.math_utils.math_equivalence import is_equiv

# Load the dataset configuration
def load_dataset_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "datasets.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["datasets"]

# Cache the configuration
_DATASET_CONFIG = None

def get_dataset_config():
    global _DATASET_CONFIG
    if _DATASET_CONFIG is None:
        _DATASET_CONFIG = load_dataset_config()
    return _DATASET_CONFIG

def load_train_dataset(config, task_id, num_tasks):
    dataset_name = config['dataset']
    dataset_config = get_dataset_config()
    if dataset_name in dataset_config:
        if dataset_name == "math" and dataset_config[dataset_name].get("math_500", True):
            # Load the 500-question subset for MATH dataset
            train_ids_path = "/n/netscratch/hankyang_lab/Lab/alex/MATH/unique_ids_train.json"
            with open(train_ids_path, "r") as f:
                train_files = json.load(f)
            
            # Construct full paths for each file
            base_path = "/n/netscratch/hankyang_lab/Lab/alex/MATH"
            train_paths = [os.path.join(base_path, file_path) for file_path in train_files]
            
            train_dataset = load_dataset(
                "json",
                data_files=train_paths,
                split="train"
            )
        else:
            train_dataset = load_dataset(
                "json",
                data_files=dataset_config[dataset_name]["train"],
                split="train"  
            )
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented.")
    # Shuffle the train dataset
    train_dataset = train_dataset.shuffle(seed=42)
    # Split the train dataset into num_tasks chunks
    train_dataset = train_dataset.shard(num_shards=num_tasks, index=task_id)
    return list(train_dataset)

def load_eval_dataset(config, task_id, num_tasks):
    dataset_name = config['dataset']
    dataset_config = get_dataset_config()
    if dataset_name in dataset_config:
        if dataset_name == "math" and dataset_config[dataset_name].get("math_500", True):
            # Load the 500-question subset for MATH dataset
            test_ids_path = "/n/netscratch/hankyang_lab/Lab/alex/MATH/unique_ids_test.json"
            with open(test_ids_path, "r") as f:
                test_files = json.load(f)
            
            # Construct full paths for each file
            base_path = "/n/netscratch/hankyang_lab/Lab/alex/MATH"
            test_paths = [os.path.join(base_path, file_path) for file_path in test_files]
            
            eval_dataset = load_dataset(
                "json",
                data_files=test_paths,
                split="train"  # Using "train" split since we're providing specific files
            )
        else:
            eval_dataset = load_dataset(
                "json",
                data_files=dataset_config[dataset_name]["test"],
            )
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented.")
    # Split the eval dataset into num_tasks chunks
    eval_dataset = eval_dataset.shard(num_shards=num_tasks, index=task_id)
    return list(eval_dataset)

def check_correct_answer(answer, correct_answer, dataset_name):
    if dataset_name == "math":
        return is_equiv(answer, correct_answer)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")