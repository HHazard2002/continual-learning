import re
import torch

def get_ll_text(text, model, tokenizer, DEVICE="cuda"):
    with torch.no_grad():
        tokenized = tokenizer(text, return_tensors="pt").to(DEVICE)
        labels = tokenized.input_ids
        output = model(**tokenized, labels=labels)
        avg_ll = -output.loss.item() # Loss is already averaged over tokens
    return avg_ll

def compute_exact_match(data):
    """
    data: list of tuples (predicted, expected)
    Returns: exact match accuracy ignoring punctuation but keeping '&'.
    """
    correct = 0
    for pred, exp in data:
        # Remove dots, commas, parentheses, hashes etc. but keep &
        pred = re.sub(r'[^\w\s&]', '', pred).strip()
        exp = re.sub(r'[^\w\s&]', '', exp).strip()

        # Handle multi-word labels
        exp_tokens = exp.split()
        pred_tokens = pred.split()

        # Take only first class up to expected length
        pred_first_class = " ".join(pred_tokens[:len(exp_tokens)])

        if pred_first_class.lower() == exp.lower():
            correct += 1

    return correct / len(data)