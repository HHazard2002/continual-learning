from datasets import load_dataset, concatenate_datasets
from pathlib import Path
import json

# Load JSON file
with open(Path("../configs/prompts.json")) as f:
    config = json.load(f)

label_mappings = config["label_mappings"]
instruction = config["instruction"]
prompt = config["prompt"]

def load_from_hf(tasks, benchmark):
    train_datasets_full = []
    test_datasets = []

    for task in tasks:
        train_datasets_full.append(load_dataset(benchmark, split=task))
        test_datasets.append(load_dataset(benchmark, split=task))

    return train_datasets_full, test_datasets

def tokenize(example, tokenizer):
    
        full_prompt = tokenizer.bos_token + example["prompt"] + " " + example["label"] + tokenizer.eos_token

        enc = tokenizer(full_prompt)

        input_len = len(tokenizer(tokenizer.bos_token + example["prompt"] + " "))

        labels = enc["input_ids"].copy()
        labels[:input_len] = [-100] * input_len

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels
        }


def prepare_datasets(train_datasets_full, tokenizer, tasks):
    train_datasets = []
    for _, dataset in enumerate(train_datasets_full):
        tokenized = dataset.map(lambda e: tokenize(e, tokenizer=tokenizer))
        tokenized = tokenized.remove_columns(['label', 'text1', 'text2'])
        print(tokenized)
        train_datasets.append(tokenized)
    return train_datasets

def prepare_mlt_dataset(train_datasets):
    joint_ds = concatenate_datasets(train_datasets)
    mlt_dataset = joint_ds.shuffle(seed=42)
    return mlt_dataset


def check_prompts_labels(tokenizer, train_datasets):
    for train_dataset in train_datasets:
        for i in range(len(train_dataset)):
            input_ids = train_dataset[i]["input_ids"]
            labels = train_dataset[i]["labels"]
            filtered_labels = [token_id for token_id in labels if token_id != -100]

            prompt_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            label_text = tokenizer.decode(filtered_labels, skip_special_tokens=True)

            print("Prompt:", prompt_text)
            print("Label:", label_text)
            print("---")