import argparse
import torch
import os
from datasets import load_dataset, load_from_disk, DatasetDict
from datetime import timedelta
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed, DummyOptim, DummyScheduler
from tqdm import tqdm
from transformers import set_seed, default_data_collator, LlamaConfig, BitsAndBytesConfig
from flash_attn.losses.cross_entropy import CrossEntropyLoss
import math
from llama_model import LlamaForCausalLM  
from peft import get_peft_model, LoraConfig


def setup_model(model_name, use_4bit=False, custom_config=None):
    """Set up the Llama model with optional 4-bit quantization and custom configuration."""
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


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb:
        import wandb
        wandb.login()

    set_seed(args.seed)
    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulate_every,
        mixed_precision="bf16",
        log_with="wandb" if args.wandb else None,
        kwargs_handlers=[timeout],
    )

    if args.wandb:
        accelerator.init_trackers(project_name=args.wandb, init_kwargs={"wandb": {"name": args.output_dir.split("/")[-1]}})

    accelerator.print(f"Total GPUs: {accelerator.num_processes}")

    try:
        train_dataset = load_dataset(args.dataset)
    except:
        train_dataset = load_from_disk(args.dataset)

    if isinstance(train_dataset, DatasetDict):
        train_dataset = train_dataset["train"]

    if "input_ids" not in train_dataset.column_names:
        raise RuntimeError("Dataset must include an `input_ids` feature")

    train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col != "input_ids"])
    train_dataset = train_dataset.shuffle(seed=args.seed)
    accelerator.print(f"Dataset Size: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        shuffle=True,
        batch_size=args.batch_size,
    )

    model = setup_model(
        model_name="TinyLlama/TinyLlama_v1.1",
        use_4bit=True,
        custom_config={"segment_size": 8, "delta_update": True, "use_cache": False}
    )

    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        r=4,
        lora_alpha=32,
        lora_dropout=0.01,
        target_modules=["k_proj", "q_proj"]
    )

    model = get_peft_model(model, peft_config=peft_config)

    if args.learning_rate != 2e-5:
        accelerator.print("Warning: Update accelerate_configs/zero3_offload.json to reflect the new learning rate.")

    optim = DummyOptim(model.parameters(), lr=args.learning_rate)
    scheduler = DummyScheduler(optim, num_training_steps=args.max_train_steps)

    model, optim, scheduler = accelerator.prepare(model, optim, scheduler)
    train_loader = accelerator.prepare(train_loader)
    model.gradient_checkpointing_enable()
    accelerator.register_for_checkpointing(scheduler)

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    model.train()

    loss_func = CrossEntropyLoss(inplace_backward=True)

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"][..., :args.seq_length + 1][..., :-1]
        target_ids = batch["input_ids"][..., :args.seq_length + 1][..., 1:]
        position_ids = torch.arange(args.seq_length).unsqueeze(0).expand(input_ids.shape[0], -1)

        local_input_ids = input_ids.to(accelerator.device)
        local_position_ids = position_ids.to(accelerator.device)
        local_target_ids = target_ids.to(accelerator.device)

        with accelerator.accumulate(model):
            logits = model(local_input_ids, position_ids=local_position_ids).logits
            loss = loss_func(logits.reshape(-1, logits.shape[-1]), local_target_ids.reshape(-1))
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                gathered_loss = accelerator.reduce(loss.clone().detach(), "mean")
                accelerator.log({"loss": gathered_loss.item(), "ppl": math.exp(gathered_loss.item())}, step=completed_steps)

            optim.step()
            scheduler.step()
            optim.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": gathered_loss.item(), "ppl": math.exp(gathered_loss.item())})
            completed_steps += 1

        if completed_steps >= args.max_train_steps:
            break

    accelerator.print("Training Finished")
    accelerator.end_training()

    if args.output_dir:
        accelerator.print(f"Saving model to {args.output_dir}")
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        accelerator.unwrap_model(model).save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=state_dict,
        )
        accelerator.print("Saving Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulate-every", type=int, default=8)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--wandb", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--rope-theta", type=float, default=100000)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset", type=str, default="emozilla/pg_books-tokenized-bos-eos-chunked-65536")
    parser.add_argument("--seq-length", type=int, default=16384)
    parser.add_argument("--parallel_mode", type=str, choices=["zigzag_ring_attn", "dist_flash_attn", "ulysses_attn", "data_parallel"])

    args = parser.parse_args()
    main(args)
