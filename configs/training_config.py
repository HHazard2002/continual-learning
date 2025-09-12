from transformers import TrainingArguments

args = TrainingArguments(
        output_dir = "llama",
        num_train_epochs=1,
        #max_steps = 1, 
        per_device_train_batch_size=1,
        #warmup_ratio = 0.3,
        logging_steps=100,
        save_strategy="epoch",
        #evaluation_strategy="epoch",
        #eval_strategy="steps",
        eval_steps=100, 
        learning_rate=1e-4, # 2e-4
        #bf16=True,
        lr_scheduler_type="constant",
        #warmup_steps=50,
        remove_unused_columns=False,
        gradient_accumulation_steps=64,
)