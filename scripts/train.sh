cl_method=""
main.py \
    --model_name "meta-llama/Llama-3.1-8B" \
    --tasks "dbpedia", "amazon", "yahoo", "ag_news" \
    --benchmark "HHazard/classical_cl-train" \
    --batch_size "1" \
    --device "cuda" \
    --lora_alpha "16" \
    --lora_rank "8" \
    --gradiet_accumulation_steps "64" \
    --learning_rate "1e-4" \
    --hf_token "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" \
    --torch_dtype "torch.bfloat16" \
    
    
    