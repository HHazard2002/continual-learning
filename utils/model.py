from peft import get_peft_model

def prepare_model(base_model, peft_config):
    model = get_peft_model(base_model, peft_config)
    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.dropout = 0.1
    return model