import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    LlamaConfig, TrainingArguments
)
from trl import SFTTrainer
import matplotlib.pyplot as plt
import numpy as np
import datetime

# ======================= CONFIG =======================
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_MODEL = "tinyllama-enhanced"
DATASET_ID = "fka/awesome-chatgpt-prompts"

# ======================= DATA PREP =======================
def format_training_data(input_text, response) -> str:
    return f"<|user|>\n{input_text}</s>\n<|assistant|>\n{response}</s>"

def prepare_training_data(dataset_id):
    data = load_dataset(dataset_id, split="train")
    data_df = data.to_pandas()
    data_df["text"] = data_df[["act", "prompt"]].apply(
        lambda x: format_training_data(x["act"], x["prompt"]), axis=1
    )
    return Dataset.from_pandas(data_df)

# ======================= MODEL SETUP =======================
def setup_model(model_name, use_4bit=False, custom_config=None):
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

    model = AutoModelForCausalLM.from_pretrained(
        model_name, config=config, quantization_config=quantization_config
    )

    # Modify self-attention gates
    for layer in model.model.layers:
        layer.self_attn.gate.data = torch.ones_like(layer.self_attn.gate.data) - 5
        layer.self_attn.gate.requires_grad = True

    return model

# ======================= TRAINING =======================
def train_model(model, tokenizer, train_data, output_dir):
    peft_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=1,
        num_train_epochs=2,
        max_steps=250,
        fp16=True
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        dataset_text_field="text",
        args=training_args,
        tokenizer=tokenizer,
        packing=False,
        max_seq_length=1024
    )

    torch.cuda.empty_cache()
    trainer.train()

    return model

# ======================= VISUALIZATION =======================
def save_heatmap_with_timestamp(data):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'heatmap_{timestamp}.png'

    data_array = np.array(data)

    plt.figure(figsize=(16, 16))
    plt.imshow(data_array, cmap='viridis', aspect='auto')
    plt.colorbar()

    for i in range(data_array.shape[0]):
        for j in range(data_array.shape[1]):
            plt.text(j, i, f'{data_array[i, j]:.2f}', ha='center', va='center', color='white')

    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.title('Heatmap of Self-Attention Gates')
    plt.savefig(filename)
    plt.close()

    return filename

# ======================= MAIN FLOW =======================
if __name__ == "__main__":
    train_data = prepare_training_data(DATASET_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    custom_config = {"segment_size": 16, "delta_update": True, "use_cache": False}
    model = setup_model(MODEL_ID, use_4bit=False, custom_config=custom_config)

    trained_model = train_model(model, tokenizer, train_data, OUTPUT_MODEL)

    gate_values = []
    for layer in trained_model.model.model.layers:
        data = layer.self_attn.gate.data.detach()
        data = torch.sigmoid(data).reshape(-1)
        gate_values.append(data.tolist())

    heatmap_file = save_heatmap_with_timestamp(gate_values)
    print(f"Heatmap saved as: {heatmap_file}")
