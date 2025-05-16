import argparse, json, os, sys, glob
from typing import List
from termcolor import colored
from datasets import load_dataset

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, repo_root)              
sys.path.insert(0, os.path.join(repo_root, 'src'))

from src.utils.gen_utils import extract_answer

def paths_for_split(root: str, split: str, dataset: str) -> List[str]:
    """
    Return a list of JSON files for `split` of `dataset`.
    Extend with more datasets as needed.
    """
    if dataset == "math":
        uid_file = os.path.join(root, f"unique_ids_{split}.json")
        if os.path.isfile(uid_file):                         # 500‑subset
            with open(uid_file) as f:
                rels = json.load(f)
            return [os.path.join(root, p) for p in rels]

        return sorted(glob.glob(os.path.join(root, "**", split, "**", "*.json"),
                                recursive=True))

    # default behavior: simple recursive glob under root/split
    pattern = os.path.join(root, "**", split, "**", "*.json")
    return sorted(glob.glob(pattern, recursive=True))


def load_dataset_with_paths(file_paths: List[str]):
    ds = load_dataset("json", data_files=file_paths, split="train")
    ds = ds.add_column("file_path", file_paths)
    return ds

def scan_split(root: str, split: str, dataset: str) -> int:
    print(colored(f"\nChecking {dataset}:{split} …", attrs=["bold"]))
    files = paths_for_split(root, split, dataset)
    if not files:
        print(colored("No files found – skipping...", "yellow"))
        return 0

    ds = load_dataset_with_paths(files)

    def mark_bad(ex):
        parsed = extract_answer(ex["solution"], dataset)
        bad = (parsed in ("", None))
        if "answer" in ex and parsed not in ("", None):
            bad |= str(parsed) != str(ex["answer"])
        ex["is_bad"] = bad
        return ex

    bad = ds.map(mark_bad).filter(lambda e: e["is_bad"])

    if len(bad) == 0:
        print(colored("  ✓ All good – no parsing failures.", "green"))
        return 0

    print(colored(f"  ✗ Found {len(bad)} problematic records:", "red"))
    for ex in bad:
        print("     •", ex["file_path"])
    return len(bad)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate dataset and print files with bad formatted answers.")
    parser.add_argument("root", help="Root directory that contains the dataset")
    parser.add_argument("--dataset", default="math",
                        help="Dataset name understood by extract_answer "
                             "(default: math)")
    parser.add_argument("--split", choices=["train", "test", "both"],
                        default="both", help="Which split(s) to check")
    args = parser.parse_args()

    splits = ["train", "test"] if args.split == "both" else [args.split]
    total_bad = sum(scan_split(args.root, sp, args.dataset) for sp in splits)

    if total_bad == 0:
        print(colored("\n✔ Dataset passes all checks.\n", "green"))
    else:
        print(colored(f"\n⚠  {total_bad} total problematic records found.\n",
                      "red"))


if __name__ == "__main__":
    main()
