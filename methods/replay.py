from transformers import Trainer
import numpy as np
import torch

class Buffer:
    def __init__(self, buffer_size, device):
        self.buffer_size = buffer_size
        self.device = device
        self.buffer = []
        self.num_seen_examples = 0
        self.num_examples_per_dom = {}

    def reservoir(self, index):
        if index < self.buffer_size:
            return index
        rand = np.random.randint(0, index + 1)
        return rand if rand < self.buffer_size else -1
        #return 1

    def add_data(self, input_ids, labels, dom):
        for i in range(len(input_ids)):
            idx = self.reservoir(self.num_seen_examples)
            input_id = input_ids[i].to(self.device)
            label = labels[i].to(self.device)
            self.num_seen_examples += 1

            if dom not in self.num_examples_per_dom:
              self.num_examples_per_dom[dom] = 0

            if idx >= 0:
                if len(self.buffer) < self.buffer_size:
                    self.buffer.append((input_id, label, dom))
                    if dom in self.num_examples_per_dom:
                        self.num_examples_per_dom[dom] += 1
                else:
                    candidate_idx = np.random.randint(0, len(self.buffer))
                    candidate = self.buffer[candidate_idx]
                    self.num_examples_per_dom[candidate[2]] -= 1
                    if dom in self.num_examples_per_dom:
                        self.num_examples_per_dom[dom] += 1
                    self.buffer[candidate_idx] = (input_id, label, dom)

    def get_data(self, size):
        if not self.buffer:
            return None
        indices = np.random.choice(len(self.buffer), size=min(size, len(self.buffer)), replace=False)
        inputs, labels, _ = zip(*[self.buffer[i] for i in indices])
        return torch.stack(inputs), torch.stack(labels)

    def is_empty(self):
        return len(self.buffer) == 0


class ReplayTrainer(Trainer):
    def __init__(self, *args, buffer=None, domain=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = buffer
        self.domain = domain
        self._micro_step = 0  # counts forward passes

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Forward on current batch
        outputs = model(**inputs)
        micro_batch_size = inputs["input_ids"].shape[0]

        # Add to buffer
        if self.buffer is not None:
            self.buffer.add_data(inputs["input_ids"], inputs["labels"], self.domain)
      
        # ---- Micro step count ----
        self._micro_step += 1
        do_replay = (self._micro_step % 4 == 0)  # every other micro step

        # ---- Replay memory (half-size batch) ----
        if do_replay and self.buffer and not self.buffer.is_empty():
            buffer_sample_size = max(1, self.args.per_device_train_batch_size // 2)
            buffer_inputs, buffer_labels = self.buffer.get_data(size=buffer_sample_size)
            replay_inputs = {"input_ids": buffer_inputs, "labels": buffer_labels}
            replay_outputs = model(**replay_inputs)
            loss = (outputs.loss + replay_outputs.loss) / 2
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
