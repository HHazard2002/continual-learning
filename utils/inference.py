
def generate_new_text(text, model, tokenizer, DEVICE):

    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    eos_token_id = tokenizer.eos_token_id
    output = model.generate(
        inputs["input_ids"],
        max_new_tokens=4,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.1,
        top_p=0.9,
        do_sample=True
    )

    #print(inputs)
    #print(output)

    input_lens = inputs["input_ids"].shape[1]
    generated_sequences = output[:, input_lens:][0]

    decoded_text = tokenizer.decode(generated_sequences, skip_special_tokens=True)

    return decoded_text