import os
import datasets
from verl.utils.hdfs_io import makedirs
import re

def extract_solution(solution_str):
    solution = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

def make_map_fn(split):
    def process_fn(example, idx):
        question_raw = example.pop("question")
        instruction_following = 'Let\'s think step by step and output the final answer after "####".'
        question = question_raw + " " + instruction_following
        answer_raw = example.pop("answer")
        solution = extract_solution(answer_raw)
        data = {
            "data_source": "openai/gsm8k",
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer_raw,
                "question": question_raw,
            },
        }
        return data
    return process_fn

if __name__ == "__main__":
    local_save_dir = os.path.expanduser("~/Projects/sdpo-test1/data/gsm8k")
    os.makedirs(local_save_dir, exist_ok=True)
    
    # Download and take only 100 examples for train, 20 for test
    dataset = datasets.load_dataset("openai/gsm8k", "main")
    train_dataset = dataset["train"].select(range(100))
    test_dataset = dataset["test"].select(range(20))
    
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
    
    print(f"Saved PoC datasets to {local_save_dir}")
