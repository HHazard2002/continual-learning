from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from methods.dual_learner import DualLearnerTrainer
from methods.ewc import EWCTrainer
from methods.replay import ReplayTrainer
from methods.surprise_replay import SurpriseReplayTrainer
from methods.buffer import Buffer
from utils.model import prepare_model
from utils.datasets import load_from_hf, prepare_datasets_llama, prepare_datasets_T5, check_prompts_labels, prepare_mlt_dataset
import torch
from utils.evaluation import evaluate_llama, evaluate_T5
from utils.script import build_cfg
from peft import LoraConfig

def resolve_dtype(x):
    if isinstance(x, torch.dtype): 
        return x
    s = str(x).lower()
    if s in {"float16", "fp16", "torch.float16"}: return torch.float16
    if s in {"bfloat16", "bf16", "torch.bfloat16"}: return torch.bfloat16
    if s in {"float32", "fp32", "torch.float32"}: return torch.float32
    return torch.float16  # sensible default

trainers = {
    'ewc': EWCTrainer,
    'replay': ReplayTrainer,
    'surprise_replay': SurpriseReplayTrainer,   # fixed key
    'dual_learner': DualLearnerTrainer,
    'mlt': Trainer
}

def pick_evaluator(model_name: str):
    if model_name.startswith("google-t5/"):
        return evaluate_T5
    if "llama" in model_name.lower() or "meta-llama" in model_name.lower():
        return evaluate_llama
    # safe default
    return evaluate_llama

def main():
    cfg = build_cfg()

    def get(k, default=None):
        return cfg.get(k, default)

    MODEL_NAME = cfg["model_name"]
    HF_TOKEN = get("hf_token")
    TORCH_DTYPE = resolve_dtype(get("torch_dtype", "float16"))
    tasks = cfg["tasks"]
    benchmark = cfg["benchmark"]
    do_check_prompts_labels = get("do_check_prompts_labels", False)
    cl_method = cfg["cl_method"]
    compute_surprise = get("compute_surprise", False)
    update_buffer = get("update_buffer")
    buffer_type = get("buffer")
    num_samples_training = get("num_samples_training")
    num_samples_eval = get("num_samples_eval")
    learning_rate = float(get("learning_rate", 2e-5))
    lora_rank = int(get("lora_rank", 16))
    lora_alpha = int(get("lora_alpha", 32))
    gradient_accumulation_steps = int(get("gradient_accumulation_steps", 1))
    batch_size = int(get("batch_size", 1))
    lambda_ema = float(get("lambda_ema", 0.999))
    eval_after_each_task = bool(get("eval_after_each_task", False))

    if buffer_type:
        buffer_size = int(get("buffer_size", 500))
        replay_frequency = int(get("replay_frequency", 2))

    DEVICE = get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {DEVICE}")

    if MODEL_NAME.startswith("google-t5/"):
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME, token=HF_TOKEN, dtype=TORCH_DTYPE
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, token=HF_TOKEN, dtype=TORCH_DTYPE
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        # safest fallback; T5 normally has a pad_token already
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    tokenizer.padding_side = "right"

    base_model.to(DEVICE)
    print(f'{MODEL_NAME} loaded and moved to device')

    # data collator
    if MODEL_NAME.startswith("google-t5/"):
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=base_model)
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        bias="none",
    )

    peft_config.task_type = "SEQ_2_SEQ_LM" if MODEL_NAME.startswith("google-t5/") else "CAUSAL_LM"

    model = prepare_model(base_model, peft_config, MODEL_NAME)
    train_datasets_full, test_datasets = load_from_hf(tasks, benchmark)
    train_datasets_full, train_datasets, test_datasets = prepare_datasets_T5(tokenizer, train_datasets_full, test_datasets, num_samples_training) if MODEL_NAME.startswith("google-t5") else prepare_datasets_llama(tokenizer, train_datasets_full, test_datasets)

    if do_check_prompts_labels:
        # pass what your function actually needs
        check_prompts_labels(tokenizer, train_datasets)

    trainer_class = trainers[cl_method]

    # --- build trainer (HF vs custom) ---
    if cl_method == 'mlt':
        training_args = TrainingArguments(
            output_dir=f"./runs/{MODEL_NAME.replace('/', '_')}",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            num_train_epochs=cfg.get("num_train_epochs", 1),
            logging_steps=cfg.get("logging_steps", 50),
            save_steps=cfg.get("save_steps", 0),
            fp16=(TORCH_DTYPE == torch.float16),
            bf16=(TORCH_DTYPE == torch.bfloat16),
            report_to="none",
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=None,
            eval_dataset=None,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    else:
        # REMOVE tokenizer=tokenizer from here:
        trainer = trainer_class(
            model=model,
            train_dataset=None,
            eval_dataset=None,
            data_collator=data_collator,
        )

    evaluate = pick_evaluator(MODEL_NAME)

    if cl_method == 'mlt':
        print(f'--------Training on all datasets---------')
        joint_ds = prepare_mlt_dataset(train_datasets)
        print(joint_ds)
        trainer.train_dataset = joint_ds
        trainer.train()
        model = trainer.model.to(DEVICE)
        model.eval()
        for test_ds, task in zip(test_datasets, tasks):
            evaluate(model, test_ds, task, tokenizer, num_samples_eval, compute_ll=False)
        model.train()
    else:
        buf = None
        if buffer_type is not None:
            buf = Buffer(
                buffer_size=buffer_size,
                device=torch.device(DEVICE),
                data_collator=data_collator,
                type=buffer_type
            )
            trainer.buffer = buf
            trainer.replay_frequency = replay_frequency
            trainer.update_buffer = update_buffer
            if update_buffer == "before" and buffer_type == "Surprise":
                buf.surprise_buffer_update(tokenizer, model, train_datasets, 0, update_buffer)

        if cl_method == "dual_learner":
            trainer.slow_model = model
            trainer.lambda_ema = lambda_ema

        for i, task in enumerate(tasks):
            print(f'--------Training on {task}---------')
            trainer.train_dataset = train_datasets[i]
            trainer.domain = i
            trainer.train()
            model = trainer.model.to(DEVICE)

            if eval_after_each_task:
                model.eval()
                for test_ds, task in zip(test_datasets, tasks):
                    evaluate(model, test_ds, task, tokenizer, num_samples_eval, compute_ll=False)
                model.train()

            if buf is not None and update_buffer in {"before", "after"} and buffer_type == "Surprise":
                buf.surprise_buffer_update(tokenizer, model, train_datasets, i, update_buffer)
            trainer.model = model

    print("--------Starting Evaluation---------")
    model.eval()
    for test_ds, task in zip(test_datasets, tasks):
        evaluate(model, test_ds, task, tokenizer, num_samples_eval, compute_ll=False)
    model.train()

if __name__ == "__main__":
    main()