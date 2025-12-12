from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import (
    DPOTrainer,
    DPOConfig,
)
from data.tldr_data_loader import load_tldr_preferences_for_trainer
import torch


def pick_device_dtype(bf16: bool) -> torch.dtype:
    if bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32

def run_dpo(args):
    dtype = pick_device_dtype(args.bf16)
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_tldr_preferences_for_trainer(tokenizer, max_prompt_tokens=args.max_prompt_tokens)

    model = AutoModelForCausalLM.from_pretrained(args.sft_model, torch_dtype=dtype)
    ref_model = AutoModelForCausalLM.from_pretrained(args.sft_model, torch_dtype=dtype)
    ref_model.requires_grad_(False)

    cfg = DPOConfig(
        output_dir=args.dpo_out,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.dpo_lr,
        num_train_epochs=args.dpo_epochs,
        max_length=args.max_seq_len,
        max_prompt_length=args.max_prompt_len_for_dpo,
        logging_steps=args.log_every,
        save_strategy="steps",
        save_steps=args.save_every,
        bf16=args.bf16,
        fp16=not args.bf16,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=cfg,
        train_dataset=ds,
        tokenizer=tokenizer,
        beta=args.dpo_beta,
    )
    trainer.train()
    trainer.save_model(args.dpo_out)
    tokenizer.save_pretrained(args.dpo_out)