from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import argparse
from configs.lora_config import peft_config
from configs.training_config import args
from methods.dual_learner import DualLearnerTrainer
from methods.ewc import EWCTrainer
from methods.replay import ReplayTrainer
from methods.surprise_replay import SurpriseReplayTrainer
from methods.buffer import Buffer
from utils.model import prepare_model
from utils.datasets import load_from_hf, prepare_datasets, check_prompts_labels
from datasets import concatenate_datasets
import torch
from utils.evaluation import evaluate_llama, evaluate_T5

# -> load the right benchmark with the right order
# -> load the right model
# -> prepare training and lora arguments
# -> start training with the right method (special loop for MLT)

trainers = {
    'ewc': EWCTrainer,
    'replay': ReplayTrainer,
    'suprise_replay': SurpriseReplayTrainer,
    'dual_learner': DualLearnerTrainer,
    'mlt': Trainer
}

evaluate_func = {
    'google-t5/t5-large': evaluate_T5,
    'meta-llama/Llama-3.1-8B': evaluate_llama
}
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("sh_file", help="Path to the .sh file")
    args = parser.parse_args()
    print(f"Shell file passed: {args.sh_file}")

    MODEL_NAME = args["model_name"]
    HF_TOKEN = args["hf_token"]
    TORCH_DTYPE = args["torch_dtype"]
    tasks = args["tasks"]
    benchmark = args["benchmark"]
    do_check_prompts_labels = args["do_check_prompts_labels"]
    cl_method = args["cl_method"]
    compute_surprise = args["compute_surprise"]
    update_buffer = args["update_buffer"]
    buffer = args["buffer"]
    num_samples_training = args["num_samples_training"]
    num_samples_eval = args["num_samples_eval"]
    learning_rate = args["learning_rate"]
    lora_rank = args["lora_rank"]
    lora_alpha = args["lora_alpha"]
    gradiet_accumulation_steps = args["gradiet_accumulation_steps"]
    batch_size = args["batch_size"]
    if buffer:
        buffer_size = args["buffer_size"]
        replay_frequency = args["replay_frequency"]

    DEVICE = args["device"]
    print(f"Running on {DEVICE}")

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN, torch_dtype=TORCH_DTYPE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model.to(DEVICE)

    print(f'{MODEL_NAME} loaded and moved to device')

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False  # mlm=False means causal language modeling
    )

    model = prepare_model(base_model, peft_config, MODEL_NAME)
    train_datasets_full, test_datasets = load_from_hf(tasks, benchmark)
    train_datasets, test_datasets = prepare_datasets(tokenizer, train_datasets_full, test_datasets)

    if do_check_prompts_labels:
        check_prompts_labels()

    trainer_class = trainers[cl_method]
    trainer = trainer_class(
        model=model,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )


    if cl_method == 'mlt':
        print(f'--------Training on all datasets---------')
        joint_ds = concatenate_datasets(train_datasets)
        train_dataset = joint_ds.shuffle(seed=42)

        trainer.train_dataset = train_dataset

        trainer.train()

        model = trainer.model
        model.to(DEVICE)

        model.eval()
        for test_ds, task in zip(test_datasets, tasks):
            evaluate(model, test_ds, task, compute_ll=True)
        model.train()

    else:
        if buffer != None:
            buffer = Buffer(buffer_size=buffer_size, device=torch.device(DEVICE), data_collator=data_collator, type=buffer)
            trainer.buffer = buffer
            trainer.replay_frequency = replay_frequency
            if update_buffer == "before":
                buffer.surprise_buffer_update(tokenizer, model, train_datasets, 0, update_buffer)
                
        for i, task in enumerate(tasks):
            print(f'--------Training on {task}---------')
            trainer.train_dataset = train_datasets[i]

            trainer.train()

            model = trainer.model
            model.to(DEVICE)

            model.eval()
            for test_ds, task in zip(test_datasets, tasks):
                evaluate(model, test_ds, task, compute_ll=True)
            model.train()

            buffer.surprise_buffer_update(tokenizer, model, train_datasets, i, update_buffer)

    print("--------Starting Evaluation---------")
    evaluate = evaluate_func[MODEL_NAME]
    model.eval()
    for test_ds, task in zip(test_datasets, tasks):
        evaluate(model, test_ds, task, compute_ll=False)
    model.train()


if __name__ == "__main__":
    main()