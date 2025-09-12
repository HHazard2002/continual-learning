import json
from pathlib import Path
from metrics import get_ll, compute_exact_match
from inference import generate_new_text
from torch.utils.data import DataLoader
import torch

# Load JSON file
with open(Path("../configs/prompts.json")) as f:
    config = json.load(f)

label_mappings = config["label_mappings"]
instruction = config["instruction"]
prompt = config["prompt"]

def evaluate_llama(model, test_ds, task, tokenizer, compute_ll=False):
      em = []
      losses = 0

      for i in range(min(100, len(test_ds))):
          sample = test_ds[i]
          text_answer = sample["label"]

          text = sample["prompt"]
          new_text = generate_new_text(text, model, tokenizer)

          if compute_ll:
            losses += get_ll(text + text_answer, model)

          predicted = new_text.strip().lower()
          expected = text_answer.strip().lower()

          em.append((predicted, expected))

      print(f"The exact match for {task} is {compute_exact_match(em):.2f}")
      print(f"The average loss per sequence for {task} is {losses / 100:.4f}")


def evaluate_T5(model, test_ds, task, tokenizer, batch_size=32, compute_ll=False, task_prefix="", DEVICE="cuda", num_eval_samples=500):
    model.eval()
    em = []
    losses = 0.0
    n_samples = min(num_eval_samples, len(test_ds))

    loader = DataLoader(range(n_samples), batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch_indices in loader:
            # Convert tensor indices -> list[int]
            idx = [int(i) for i in batch_indices]

            batch = test_ds[idx]   # dict of lists
            batch_prompts = [task_prefix + p for p in batch["prompt"]]
            batch_labels  = batch["label"]

            # Tokenize & generate in batch
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(DEVICE)
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
            )
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)


            em.extend(zip(preds, batch_labels))

    print(f"The exact match for {task} is {compute_exact_match(em):.2f}")
    if compute_ll:
        avg_loss = losses / min(100, n_samples)
        print(f"The average loss per sequence for {task} is {avg_loss:.4f}")
