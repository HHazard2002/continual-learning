import torch
import numpy as np
from utils.metrics import get_lls



class Buffer:
    def __init__(self, buffer_size=500, device='cuda', type="Reservoir"):
        self.buffer_size = buffer_size
        self.device = device
        self.buffer = []
        self.num_seen_examples = 0
        self.num_examples_per_dom = {}
        self.type = type

    def reservoir(self, index):
        if index < self.buffer_size:
            return index
        rand = np.random.randint(0, index + 1)
        return rand if rand < self.buffer_size else -1


    def add_data(self, input_ids, labels, dom):
        for i in range(len(input_ids)):
            idx = self.reservoir(self.num_seen_examples) if self.type == 'Reservoir' else 1
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
        print(f"Actually deleted {actually_removed} from domain {j}")
        self.buffer = new_buffer

    def get_data(self, size):
        if not self.buffer:
            return None
        indices = np.random.choice(len(self.buffer), size=min(size, len(self.buffer)), replace=False)
        inputs, labels, _ = zip(*[self.buffer[i] for i in indices])
        return torch.stack(inputs), torch.stack(labels)

    def is_empty(self):
        return len(self.buffer) == 0
    
    def surprise_buffer_update(self, tokenizer, model, train_datasets, i, update_buffer):
        for j in range(len(train_datasets)):
            if  not self.is_empty() and ((j < i and i < 14 and update_buffer=="after") or (j <= i and i < 14 and update_buffer=="before")):

                curr_domain = [self.buffer[x] for x in range(len(self.buffer)) if self.buffer[x][3] == j]

                curr_domain = sorted(curr_domain, key=lambda x: x[2], reverse=False)

                num_to_remove = self.num_examples_per_dom[j] - (self.buffer_size // (i + 2)) if update_buffer=="before" else self.num_examples_per_dom[j] - (self.buffer_size // (i + 1))

                if num_to_remove > 0:
                    print(f"deleting {num_to_remove} from domain {j}")
                    print(f'Length of delete_data: {len([x[0] for x in curr_domain[:num_to_remove]])}')
                    self.remove_data([x[0] for x in curr_domain[:num_to_remove]], dom=j)

            if  self.is_empty() or (j == i and i < 14 and update_buffer=="after") or (j == (i+1) and i < 14 and update_buffer=="before"):
        
                lls, seq_to_labels = get_lls(
                    train_datasets[j]['input_ids'],
                    train_datasets[j]["labels"],
                    model
                )

                surprise = sorted(lls.items(), key=lambda x: x[1], reverse=True)
                if self.is_empty:
                    num_to_add = self.buffer_size
                elif update_buffer == "before":
                    num_to_add = self.buffer_size // (i + 2) 
                else:
                    num_to_add = self.buffer_size // (i + 1)
                input_ids_batch = []
                labels_batch = []
                surprise_scores_batch = []

                for k, score in surprise[:num_to_add]:
                    ids = list(k)
                    if ids[-1] != tokenizer.eos_token_id:
                        ids.append(tokenizer.eos_token_id)
                    input_ids_batch.append(torch.tensor(ids, dtype=torch.long))
                    labels_batch.append(torch.tensor(seq_to_labels[k], dtype=torch.long))  # correct label sequence
                    surprise_scores_batch.append(score)

                self.add_data(
                    input_ids_batch,
                    labels_batch,
                    surprise_scores_batch,
                    j
                )
