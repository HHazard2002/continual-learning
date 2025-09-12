from transformers import Trainer
import torch
import numpy as np

class Buffer:
    def __init__(self, buffer_size, device):
        self.buffer_size = buffer_size
        self.device = device
        self.buffer = []
        self.num_seen_examples = 0
        self.num_examples_per_dom = {}

    def add_data(self, input_ids, labels, surprise_scores, dom):
        for i in range(len(input_ids)):
            input_id = input_ids[i]
            label = labels[i]
            surprise = surprise_scores[i]

            # If tuple → convert to tensor
            if isinstance(input_id, tuple):
                input_id = torch.tensor(input_id, dtype=torch.long)
            if isinstance(label, tuple):
                label = torch.tensor(label, dtype=torch.long)
            if not torch.is_tensor(surprise):
                surprise = torch.tensor(surprise)

            input_id = input_id.to(self.device)
            label = label.to(self.device)
            surprise = surprise.to(self.device)


            self.num_seen_examples += 1

            if dom not in self.num_examples_per_dom:
              self.num_examples_per_dom[dom] = 0

            if len(self.buffer) < self.buffer_size:
                    self.buffer.append((input_id, label, surprise, dom))
                    if dom in self.num_examples_per_dom:
                        self.num_examples_per_dom[dom] += 1

    def remove_data(self, input_ids, dom):
        to_remove = set(ids for ids in input_ids)
        new_buffer = []
        actually_removed = 0  # Count what we actually remove

        for sample in self.buffer:
            if sample is None:
                continue
            input_ids_tuple = tuple(sample[0].cpu().tolist())
            if sample[0] in to_remove:
                actually_removed += 1  # Increment actual count
                del sample
            else:
                new_buffer.append(sample)

        # Use actual count, not requested count
        self.num_examples_per_dom[dom] -= actually_removed
        print(f"Actually deleted {actually_removed} from domain {dom}")
        self.buffer = new_buffer

                
    def get_data(self, size):
        if not self.buffer:
            return None
        indices = np.random.choice(len(self.buffer), size=min(size, len(self.buffer)), replace=False)
        inputs, labels, _, _ = zip(*[self.buffer[i] for i in indices])
        return inputs, labels


    def is_empty(self):
        return len(self.buffer) == 0

class DualLearnerTrainer(Trainer):
    def __init__(self, *args, slow_model=None, buffer=None, param_buffer=None, domain=None, lambda_ema=0.99, gamma=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.slow_model = slow_model
        self.buffer = buffer
        self.lambda_ema = lambda_ema
        self.domain = domain

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        batch_size = inputs["input_ids"].shape[0]

        # Replay from buffer
        if self.buffer and not self.buffer.is_empty():
            buffer_inputs, buffer_labels = self.buffer.get_data(size=batch_size // 2)

            replay_inputs = {
                "input_ids": buffer_inputs,
                "labels": buffer_labels,
            }
            
            replay_outputs = model(**replay_inputs)
            loss = (2 * outputs.loss + replay_outputs.loss) / 3
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    def training_step(
        self,
        model,
        inputs,
        num_items_in_batch: int = None,          # <— add this
    ):
        # Delegate to the original Trainer.training_step
        loss = super().training_step(model, inputs, num_items_in_batch)

        # Update slow weights (EMA) and buffer
        self.update_slow_model(model)

        return loss

    def update_slow_model(self, fast_model):
        current_step = self.state.global_step
        fast_param = fast_model.parameters()

        for p_slow, p_fast in zip(self.slow_model.parameters(), fast_param):
            p_slow.data = self.lambda_ema * p_slow.data + (1 - self.lambda_ema) * p_fast.data

    '''
    def update_slow_model(self, fast_model):
    current_step = max(1, self.state.global_step)  # avoid divide by zero
    # Step-dependent EMA decay (increases with training steps)
    lambda_ema = min(1 - 1 / (current_step + 1), self.lambda_ema)  

    with torch.no_grad():
        for p_slow, p_fast in zip(self.slow_model.parameters(), fast_model.parameters()):
            p_slow.mul_(lambda_ema).add_(p_fast, alpha=(1.0 - lambda_ema))

    '''
