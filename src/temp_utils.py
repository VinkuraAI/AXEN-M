import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTTrainer

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_MODEL = "tinyllama-enhanced"
DATASET_ID = "Anthropic/hh-rlhf"

def format_training_data(input_text, response) -> str:
    """Format data for training."""
    return f"<|user|>\n{input_text}</s>\n<|assistant|>\n{response}</s>"

def prepare_training_data(dataset_id):
    """Load and format dataset for training."""
    data = load_dataset(dataset_id, split="train")
    data_df = data.to_pandas()
    data_df["text"] = data_df[["chosen", "rejected"]].apply(
        lambda x: format_training_data(x["chosen"], x["rejected"]), axis=1
    )
    return Dataset.from_pandas(data_df)

# Prepare data
data = prepare_training_data(DATASET_ID)
print(data[0])

def load_model_and_tokenizer(model_id):
    """Load model and tokenizer with 4-bit quantization."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    return model, tokenizer

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer(MODEL_ID)

# LoRA config
peft_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

# Training arguments
training_arguments = TrainingArguments(
    output_dir=OUTPUT_MODEL,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    logging_steps=10,
    num_train_epochs=2,
    max_steps=250,
    fp16=True
)

# Trainer setup
trainer = SFTTrainer(
    model=model,
    train_dataset=data,
    peft_config=peft_config,
    dataset_text_field="text",
    args=training_arguments,
    tokenizer=tokenizer,
    packing=False,
    max_seq_length=1024
)

# Clear cache and train
torch.cuda.empty_cache()
trainer.train()
