import argparse

def build_cfg():
    p = argparse.ArgumentParser()
    # core model / method
    p.add_argument("--model_name", required=True)
    p.add_argument("--cl_method", required=True, choices=["mlt", "ewc", "replay", "surprise_replay", "dual_learner"])
    p.add_argument("--tasks", required=True, help='Space- or comma-separated list, e.g. "dbpedia amazon yahoo ag_news"')
    p.add_argument("--benchmark", required=True)
    # training knobs
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--torch_dtype", default="float16")
    p.add_argument("--num_samples_eval", type=int, default=None)
    p.add_argument("--num_samples_training", type=int, default=None)
    p.add_argument("--device", default=None)  # auto if None
    # LoRA
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_rank", type=int, default=16)
    # buffer / replay
    p.add_argument("--buffer", default=None)
    p.add_argument("--buffer_size", type=int, default=0)
    p.add_argument("--replay_frequency", type=int, default=1)
    p.add_argument("--update_buffer", default=None, choices=[None, "before", "after"])
    p.add_argument("--compute_surprise", type=lambda x: str(x).lower() in {"1","true","yes"}, default=False)
    # misc
    p.add_argument("--hf_token", default=None)
    p.add_argument("--do_check_prompts_labels", type=lambda x: str(x).lower() in {"1","true","yes"}, default=False)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--save_steps", type=int, default=0)
    p.add_argument("--lambda_ema", type=float, default=0.999)
    p.add_argument("--eval_after_each_task", type=lambda x: str(x).lower() in {"1","true","yes"}, default=True)

    args = p.parse_args()
    cfg = vars(args)

    # normalize tasks into a list
    t = cfg["tasks"].replace(",", " ").split()
    cfg["tasks"] = t
    return cfg