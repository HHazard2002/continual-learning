from transformers import Trainer
import numpy as np
import torch


class ReplayTrainer(Trainer):
    def __init__(self, *args, buffer=None, domain=None, replay_frequency=2 **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = buffer
        self.domain = domain
        self._micro_step = 0  # counts forward passes
        self.replay_frequency = replay_frequency

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Forward on current batch
        outputs = model(**inputs)
        micro_batch_size = inputs["input_ids"].shape[0]

        # Add to buffer
        if self.buffer is not None:
            self.buffer.add_data(inputs["input_ids"], inputs["labels"], self.domain)
      
        # ---- Micro step count ----
        self._micro_step += 1
        do_replay = (self._micro_step % self.replay_frequency == 0)  # every other micro step

        # ---- Replay memory (half-size batch) ----
        if do_replay and self.buffer and not self.buffer.is_empty():
            buffer_sample_size = max(1, self.args.per_device_train_batch_size)
            buffer_inputs, buffer_labels = self.buffer.get_data(size=buffer_sample_size)
            replay_inputs = {"input_ids": buffer_inputs, "labels": buffer_labels}
            replay_outputs = model(**replay_inputs)
            loss = (outputs.loss + replay_outputs.loss) / 2
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
