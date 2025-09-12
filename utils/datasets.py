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

def tokenize_t5(ex, tokenizer, task_prefix=""):
    source = (task_prefix + ex["prompt"]).strip()
    target = ex["label"]

    model_inputs = tokenizer(source)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(target)["input_ids"]

    # Fix edge cases: labels must be a non-empty list[int]
    if not isinstance(labels, list):
        labels = list(labels)  # in case something returned a numpy array/torch tensor
    if len(labels) == 0:
        labels = [tokenizer.pad_token_id]  # avoid zero-length labels

    model_inputs["labels"] = labels
    return model_inputs

def subsample(ex, MAX_SRC=512, MAX_TGT=256):
        return len(ex["input_ids"]) <= MAX_SRC and len(ex["labels"]) <= MAX_TGT

def prepare_datasets(tokenizer, train_datasets_full, test_datasets, MAX_SRC=512, MAX_TGT=256):
    for i in range(len(train_datasets_full)):
        train_datasets_full[i] = train_datasets_full[i].map(lambda e: tokenize_t5(e, tokenizer=tokenizer))
        train_datasets_full[i] = train_datasets_full[i].filter(subsample, batched=False)

        test_datasets[i] = test_datasets[i].map(tokenize_t5)
        test_datasets[i]  = test_datasets[i].filter(subsample, batched=False)
    return train_datasets_full, test_datasets

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