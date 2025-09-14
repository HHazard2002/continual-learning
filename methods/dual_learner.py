from transformers import Trainer
import torch
import numpy as np

class DualLearnerTrainer(Trainer):
    def __init__(self, *args, slow_model=None, buffer=None, domain=None, lambda_ema=0.99, replay_frequency=2, update_buffer='online', **kwargs):
        super().__init__(*args, **kwargs)
        self.slow_model = slow_model
        self.buffer = buffer
        self.lambda_ema = lambda_ema
        self.domain = domain
        self._micro_step = 0
        self.replay_frequency = replay_frequency
        self.update_buffer = update_buffer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        batch_size = inputs["input_ids"].shape[0]

        if self.update_buffer == "online" and self.buffer is not None:
            self.buffer.add_data(inputs["input_ids"], inputs["labels"], self.domain)


        self._micro_step += 1
        do_replay = (self._micro_step % self.replay_frequency == 0)

        # ---- Replay memory (half-size batch) ----
        if do_replay and self.buffer and not self.buffer.is_empty():
            #print("replayed")
            buffer_sample_size = max(1, self.args.per_device_train_batch_size)
            # get a collated batch dict (CPU tensors)
            buffer_batch = self.buffer.get_data(size=buffer_sample_size)
            # move to device since we're calling the model directly
            buffer_batch = {k: v.to(model.device) for k, v in buffer_batch.items()}
            replay_outputs = model(**buffer_batch)
            loss = (outputs.loss + replay_outputs.loss) / 2
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

