from transformers import Trainer
import torch
import numpy as np
from torch.utils.data import DataLoader

def compute_fisher(
    model,
    tokenized_dataset,
    device,
    data_collator,
    accumulation_steps=8,
    num_samples=480
):
    model.eval()
    loader = DataLoader(
        tokenized_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=data_collator,
    )

    # only keep fisher for LoRA parameters
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if "lora" in n}

    model.zero_grad()
    for i, batch in enumerate(loader):
        if i >= num_samples:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss
        (loss / accumulation_steps).backward()

        if (i + 1) % accumulation_steps == 0:
            for n, p in model.named_parameters():
                if "lora" in n and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)
            model.zero_grad()

    # handle last incomplete accumulation
    if (i + 1) % accumulation_steps != 0:
        for n, p in model.named_parameters():
            if "lora" in n and p.grad is not None:
                fisher[n] += p.grad.data.pow(2)
        model.zero_grad()

    # normalize
    fisher = {n: f / (num_samples / accumulation_steps) for n, f in fisher.items()}

    # move fisher to CPU to save GPU memory
    fisher = {n: f.cpu() for n, f in fisher.items()}
    return fisher

class EWCTrainer(Trainer):
    def __init__(self, ewc_lambda, task_fishers, task_params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ewc_lambda = ewc_lambda
        self.task_fishers = task_fishers  # list of dicts
        self.task_params = task_params    # list of dicts

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch: int = None):
        outputs = model(**inputs)
        loss = outputs.loss

        # EWC penalty: sum over all past tasks
        ewc_penalty = 0.0
        for fisher, params in zip(self.task_fishers, self.task_params):
            for n, p in model.named_parameters():
                if n in fisher:
                    diff = p - params[n].to(p.device)
                    ewc_penalty += (fisher[n].to(p.device) * diff.pow(2)).sum()

        loss = loss + (self.ewc_lambda / 2) * ewc_penalty
        return (loss, outputs) if return_outputs else loss