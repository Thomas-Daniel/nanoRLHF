import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
    set_seed,
)
from accelerate import Accelerator

from data.tldr_data_loader import load_tldr_preferences_for_trainer


def pick_device_dtype(bf16: bool) -> torch.dtype:
    if bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class RewardModel(nn.Module):
    """
    Reward model using an encoder-style backbone with a scalar head.
    Implemented via AutoModelForSequenceClassification(num_labels=1).
    """
    def __init__(self, model_name_or_path: str, dtype: torch.dtype, load_in_4bit: bool = False):
        super().__init__()
        kwargs = dict(torch_dtype=dtype)
        if load_in_4bit:
            kwargs.update(dict(load_in_4bit=True, device_map="auto"))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=1,
            **kwargs,
        )

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits.squeeze(-1)


def rm_pairwise_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    return -torch.nn.functional.logsigmoid(r_chosen - r_rejected).mean()


def run_rm(args):
    accelerator = Accelerator(mixed_precision="bf16" if args.bf16 else "fp16")
    set_seed(args.seed)
    dtype = pick_device_dtype(args.bf16)
    tokenizer = AutoTokenizer.from_pretrained(args.rm_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_tldr_preferences_for_trainer()
    rm = RewardModel(args.rm_model, dtype=dtype, load_in_4bit=args.load_in_4bit)
    optimizer = torch.optim.AdamW(rm.parameters(), lr=args.rm_lr)
    dl = DataLoader(ds, batch_size=args.rm_batch_size, shuffle=True)
    rm, optimizer, dl = accelerator.prepare(rm, optimizer, dl)

    total_steps = len(dl)
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    rm.train()
    for step, ex in enumerate(dl):
        prompts = ex["prompt"]
        chosen = ex["chosen"]
        rejected = ex["rejected"]

        enc_c = tokenizer(
            [p + " " + c for p, c in zip(prompts, chosen)],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=args.max_seq_len,
        )
        enc_r = tokenizer(
            [p + " " + r for p, r in zip(prompts, rejected)],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=args.max_seq_len,
        )

        enc_c = {k: v.to(accelerator.device) for k, v in enc_c.items()}
        enc_r = {k: v.to(accelerator.device) for k, v in enc_r.items()}

        r_c = rm(enc_c["input_ids"], enc_c["attention_mask"])
        r_r = rm(enc_r["input_ids"], enc_r["attention_mask"])

        loss = rm_pairwise_loss(r_c, r_r)

        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if accelerator.is_main_process and step % args.log_every == 0:
            accelerator.print(f"[RM] step {step}/{total_steps} loss={loss.item():.4f}")

    if accelerator.is_main_process:
        ensure_dir(args.rm_out)
        accelerator.print(f"Saving reward model to: {args.rm_out}")
        unwrapped = accelerator.unwrap_model(rm).model
        unwrapped.save_pretrained(args.rm_out)
        tokenizer.save_pretrained(args.rm_out)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_model", type=str, default=None)
    p.add_argument(
        "--rm_model",
        type=str,
        default="OpenAssistant/reward-model-deberta-v3-large-v2",
        help="Backbone for the reward model (must support AutoModelForSequenceClassification).",
    )
    p.add_argument("--rm_out", type=str, default="checkpoints/rm_tldr")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")

    p.add_argument("--rm_batch_size", type=int, default=8)
    p.add_argument("--rm_lr", type=float, default=1e-5)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--log_every", type=int, default=25)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_rm(args)
