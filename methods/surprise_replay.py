from transformers import Trainer
import numpy as np
import torch


def _seq_to_key(t: torch.Tensor, PAD_ID) -> tuple[int, ...]:
    """Convert to hashable key after removing pad tokens on either end."""
    # 1‑D tensor expected here
    # Drop left & right padding; middle padding cannot appear in causal LM.
    if PAD_ID is not None:
        t = t[t != PAD_ID]
    return tuple(t.tolist())

def get_ll(input_ids, labels, model):
    with torch.no_grad():
        # Convert to tensors if not already
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels).unsqueeze(0).to(model.device)

        sequence_labels = input_ids.clone()
        sequence_labels[:, 0] = -100  # mask first token

        out = model(input_ids=input_ids, labels=sequence_labels)
        per_token_nll = out.loss.item()

    return per_token_nll


def get_lls(inputs, labels, model, PAD_ID):
    avg_lls = {}
    seq_to_labels = {}  # key → original labels

    for input_ids, label in zip(inputs, labels):
        ll = get_ll(input_ids, label, model)
        key = _seq_to_key(torch.tensor(input_ids), PAD_ID=PAD_ID)
        avg_lls[key] = ll
        seq_to_labels[key] = label  # store actual labels for this key

    return avg_lls, seq_to_labels

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
        return torch.stack(inputs), torch.stack(labels)

    def is_empty(self):
        return len(self.buffer) == 0

class SurpriseReplayTrainer(Trainer):
    def __init__(self, *args, buffer=None, domain=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = buffer
        self.domain = domain
        self._micro_step = 0  # counts forward passes

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        micro_batch_size = inputs["input_ids"].shape[0]

        self._micro_step += 1
        do_replay = (self._micro_step % 4 == 0)  # every other micro step

        if do_replay and self.buffer and not self.buffer.is_empty():
            buffer_sample_size = max(1, self.args.per_device_train_batch_size // 2)
            buffer_inputs, buffer_labels = self.buffer.get_data(size=buffer_sample_size)
            replay_inputs = {"input_ids": buffer_inputs, "labels": buffer_labels}
            replay_outputs = model(**replay_inputs)
            loss = (outputs.loss + replay_outputs.loss) / 2
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    