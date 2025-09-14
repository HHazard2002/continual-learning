#!/usr/bin/env bash
set -euo pipefail

python3 ./main.py \
  --model_name="google-t5/t5-large" \
  --cl_method="mlt" \
  --tasks="dbpedia amazon yahoo ag_news" \
  --benchmark="HHazard/large-number-of-tasks" \
  --batch_size=1 \
  --gradient_accumulation_steps=64 \
  --learning_rate=1e-3 \
  --torch_dtype=bfloat16 \
  --num_samples_eval=2 \
  --num_samples_training=5 \
  --buffer="Reservoir" \
  --update_buffer="before" \
  --compute_surprise=true \
  --cl_method="replay" \
  --device="mps" \
  --lora_alpha=32 \
  --lora_rank=8 \
  --do_check_prompts_labels=true \
  --eval_after_each_task=false