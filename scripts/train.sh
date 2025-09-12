cl_method=""
main.py \
    --model_name  "google-t5/t5-large"\
    --tasks "dbpedia", "amazon", "yahoo", "ag_news" \
    --benchmark "HHazard/classical_cl-train" \
    --batch_size "1" \
    --gradiet_accumulation_steps "64" \
    --device "cuda" \
    --lora_alpha "32" \
    --lora_rank "8" \
    --learning_rate "1e-3" \
    --torch_dtype "torch.bfloat16" \
    --num_samples_eval "500" \
    --num_samples_training "1000" \

    
    