import os
import yaml
import json
from datasets import load_dataset as hf_load_dataset, Dataset
from src.utils.util import extract_answer
from termcolor import colored

class DatasetManager:
    def __init__(self, config: dict, task_id: int, num_shards: int):
        self.config = config
        self.dataset_name = config['base_config']['dataset']
        self._load_dataset_config()
        assert self.dataset_name in self.dataset_config, f"Dataset {self.dataset_name} not found in config"
        self.num_shards = num_shards
        self.task_id = task_id

    def save_dataset(self, filepath = None):
        if filepath is None:
            filepath =  f"{self.config['base_config']['output_base_path']}/{self.config['base_config']['dataset']}/gen/{self.config['base_config']['output_file_name'].format(task_id=self.task_id)}"
        with open(filepath, "w") as f:
            json.dump(self.dataset, f)
        print(colored(f"Saved dataset to {filepath}", "yellow"))
    
    def extract_ground_truth(self):
        for d in self.dataset:
            if 'gt_answer' not in d:
                d['gt_answer'] = extract_answer(d['solution'], self.dataset_name)
        return self.dataset

    def load_dataset(self, split: str, eval: bool = False):
        if eval:
            self.dataset = self.load_pre_generated_dataset()
        elif split == "train":
            self.dataset = self.load_train_dataset(self.dataset_name)
        elif split == "test":
            self.dataset = self.load_eval_dataset(self.dataset_name)
        else:
            raise ValueError(f"Invalid split: {split}")
        self.dataset_size = len(self.dataset)
        
    def load_train_dataset(self, dataset_name: str):
        if dataset_name == "math_500":
            # Load the 500-question subset for MATH dataset
            train_ids_path = self.dataset_config[dataset_name]['train']
            with open(train_ids_path, "r") as f:
                train_files = json.load(f)
            
            # Construct full paths for each file
            base_path = "/n/netscratch/ydu_lab/Lab/alex/MATH"
            train_paths = [os.path.join(base_path, file_path) for file_path in train_files]
            
            train_dataset = hf_load_dataset(
                "json",
                data_files=train_paths,
                split="train"
            )
        else:
            train_dataset = hf_load_dataset(
                "json",
                data_files=self.dataset_config[dataset_name]["train"],
                split="train"  
            )
        # Shuffle the train dataset
        train_dataset = train_dataset.shuffle(seed=42)
        # Split the train dataset into num_tasks chunks
        train_dataset = train_dataset.shard(num_shards=self.num_shards, index=self.task_id)
        return list(train_dataset)
    
    def load_eval_dataset(self, dataset_name: str):
        if dataset_name == "math_500":
            # Load the 500-question subset for MATH dataset
            test_ids_path = self.dataset_config[dataset_name]['test']
            with open(test_ids_path, "r") as f:
                test_files = json.load(f)
            
            # Construct full paths for each file
            base_path = "/n/netscratch/ydu_lab/Lab/alex/MATH"
            test_paths = [os.path.join(base_path, file_path) for file_path in test_files]
            
            eval_dataset = hf_load_dataset(
                "json",
                data_files=test_paths,
                split="train"  # Using "train" split since we're providing specific files
            )
        else:
            eval_dataset = hf_load_dataset(
                "json",
                data_files=self.dataset_config[dataset_name]["test"],
            )
        eval_dataset = eval_dataset.shard(num_shards=self.num_shards, index=self.task_id)
        return list(eval_dataset)
    
    def load_pre_generated_dataset(self):
        solutions_dir = self.config['base_config']['solutions_dir']
        solutions_file_name = self.config['base_config']['solutions_file_name'].format(task_id=self.task_id)
        solutions_path = os.path.join(solutions_dir, solutions_file_name)
        with open(solutions_path, "r") as f:
            dataset_list = json.load(f)

        # Convert list back to Dataset object to enable sharding
        dataset = Dataset.from_list(dataset_list)
        # Split the dataset into num_tasks chunks
        dataset = dataset.shard(num_shards=self.num_shards, index=self.task_id)
        return list(dataset)

    def _load_dataset_config(self):
        dataset_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "datasets.yaml")
        with open(dataset_config_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
        self.dataset_config = dataset_config['datasets']
    

