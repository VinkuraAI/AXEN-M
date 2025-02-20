import torch
from transformers import AutoTokenizer, LlamaConfig, BitsAndBytesConfig
from modeling_llama import LlamaForCausalLM
from peft import get_peft_model, LoraConfig


def setup_model(model_name, use_4bit=False, custom_config=None):
    """Set up the Llama model with optional 4-bit quantization."""
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    config = LlamaConfig.from_pretrained(model_name, attn_implementation="eager")
    if custom_config:
        config.__dict__.update(custom_config)

    model = LlamaForCausalLM.from_pretrained(
        model_name,
        config=config,
        quantization_config=quantization_config
    )

    for layer in model.model.layers:
        layer.self_attn.gate.data = torch.ones_like(layer.self_attn.gate.data) - 6
        layer.self_attn.gate.requires_grad = True

    return model


def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7, top_k=50):
    """Generate text using the model with configurable sampling parameters."""
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        use_cache=False
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    model_name = "TinyLlama/TinyLlama_v1.1"
    custom_config = {"segment_size": 8, "delta_update": True, "use_cache": False}
    use_4bit = True

    # Set up the model
    model = setup_model(model_name, use_4bit=use_4bit, custom_config=custom_config)

    # Apply LoRA configuration
    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        r=4,
        lora_alpha=32,
        lora_dropout=0.01,
        target_modules=['k_proj', 'q_proj']
    )
    model = get_peft_model(model, peft_config=peft_config)
    model = model.cuda()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prompt and text generation
    prompt = "September 2007 In high school I decided I was going to study philosophy in college."
    generated_text = generate_text(
        model, tokenizer, prompt, max_new_tokens=100, temperature=0.9, top_k=40
    )

    print("Generated Text:\n", generated_text)
