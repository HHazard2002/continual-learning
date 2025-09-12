from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import argparse
from configs.lora_config import peft_config
from configs.training_config import args
from methods.dual_learner import DualLearnerTrainer
from methods.ewc import EWCTrainer
from methods.replay import ReplayTrainer
from methods.surprise_replay import SurpriseReplayTrainer
from utils.model import prepare_model
from utils.datasets import load_from_hf, prepare_datasets, check_prompts_labels
from datasets import concatenate_datasets
from utils.evaluation import evaluate

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

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("sh_file", help="Path to the .sh file")
    args = parser.parse_args()
    print(f"Shell file passed: {args.sh_file}")

    DEVICE = args["device"]
    print(DEVICE)

    MODEL_NAME = args["model_name"]
    HF_TOKEN = args["hf_token"]
    TORCH_DTYPE = args["torch_dtype"]
    tasks = args["tasks"]
    benchmark = args["benchmark"]
    do_check_prompts_labels = args["do_check_prompts_labels"]
    cl_method = args["cl_method"]

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN, torch_dtype=TORCH_DTYPE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model.to(DEVICE)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False  # mlm=False means causal language modeling
    )

    model = prepare_model(base_model, peft_config)
    train_datasets_full = load_from_hf(tasks, benchmark)
    train_datasets, test_datasets = prepare_datasets(train_datasets_full)

    if do_check_prompts_labels:
        check_prompts_labels()

    trainer_class = trainers[cl_method]
    trainer = trainer_class(
        model=model,
        train_dataset=None,  # will be replaced in loop
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
        if cl_method != "ewc":
            buffer = 
            trainer.buffer = buffer
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
   

if __name__ == "__main__":
    main()