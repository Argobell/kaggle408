from unsloth import FastModel
import torch
from PIL import Image
from swanlab.integration.transformers import SwanLabCallback
from unsloth import PatchDPOTrainer
from trl import DPOTrainer, DPOConfig
import argparse
from datasets import load_dataset
from pathlib import Path
import os

def main(args):

    PatchDPOTrainer()

    swanlab_callback = SwanLabCallback(
        project="kaggle408-dpo",
        experiment_name="gemma3n-mutlti-dpo",
    )

    # Load model and tokenizer
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_name_or_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        attn_implementation="eager",
        use_gradient_checkpointing = "unsloth",
    )

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=args.tune_vision,
        finetune_language_layers=args.tune_language_layers,
        finetune_attention_modules=args.tune_attention_modules,
        finetune_mlp_modules=args.tune_mlp_modules,
        lora_dropout=args.lora_dropout,
        r=args.r,
        lora_alpha=args.alpha,
        bias="none",
        random_state=3407,
        target_modules="all-linear",
        modules_to_save=[
            "lm_head",
            "embed_tokens",
        ],
    )

    def format(example):
        prompt = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": example["question"]}],
            },
        ]
        chosen = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example["chosen"]}],
            },
        ]
        rejected = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example["rejected"]}],
            },
        ]

        max_size = max(tokenizer.image_processor.size.values())
        example["image"].thumbnail((max_size, max_size))

        if isinstance(example["image"], Image.Image) and example["image"].mode != "RGB":
            example["image"] = example["image"].convert("RGB")

        return {"images": [example["image"]], "prompt": prompt, "chosen": chosen, "rejected": rejected}
    
    train_set = load_dataset(args.dataset_name,split="train[:18%]")
    eval_set = load_dataset(args.dataset_name,split="train[18%:20%]")

    train_set = train_set.map(format, remove_columns=train_set.column_names)
    eval_set = eval_set.map(format, remove_columns=eval_set.column_names)

    trainer = DPOTrainer(
        model = model,
        ref_model = None,
        callbacks=[swanlab_callback],
        args = DPOConfig(
            per_device_train_batch_size = args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_checkpointing=args.gradient_checkpointing,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            eval_strategy=args.eval_strategy,
            save_steps=args.save_steps,
            save_strategy=args.save_strategy,
            optim=args.optim,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            seed = 3407,
            output_dir=args.output_dir,
            dataloader_num_workers=args.dataloader_num_workers,
            dataset_num_proc=args.dataset_num_proc,
        ),
        processing_class= tokenizer.tokenizer,
        beta = 0.1,
        train_dataset = train_set,
        eval_dataset = eval_set,
        max_length = args.max_seq_length,
        max_prompt_length = args.max_prompt_length,
    )

    trainer_stats = trainer.train()

    torch.cuda.empty_cache()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.save_merged:
        parent_dir = os.path.dirname(args.output_dir)
        new_subdir = os.path.join(parent_dir, "gek_e2b_dpo")
        os.makedirs(new_subdir, exist_ok=True)
        model.save_pretrained_merged(new_subdir,tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a vision model with Unsloth using DPO")

    # Model and PEFT parameters
    parser.add_argument("--model_name_or_path", type=str, default="unsloth/gemma-3n-E2B-it", help="Model name or path.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--max_prompt_length", type=int, default=1024, help="Maximum prompt length.")
    parser.add_argument("--load_in_4bit", action='store_true', help="Load model in 4-bit precision.")
    parser.add_argument("--tune_vision", action='store_true', help="Fine-tune vision layers.")
    parser.add_argument("--tune_language_layers", action='store_true', help="Fine-tune language layers.")
    parser.add_argument("--tune_attention_modules", action='store_true', help="Fine-tune attention modules.")
    parser.add_argument("--tune_mlp_modules", action='store_true', help="Fine-tune MLP modules.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--r", type=int, default=16, help="LoRA r parameter.")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha parameter.")

    # Dataset parameters
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset on Hugging Face Hub.")
    parser.add_argument("--dataset_num_proc", type=int, default=4, help="Number of processes for dataset mapping.")

    # Training parameters
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Training batch size per device.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Evaluation batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--gradient_checkpointing", action='store_true', help="Enable gradient checkpointing.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio.")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate.")

    # Logging and saving
    parser.add_argument("--logging_steps", type=int, default=5, help="Log every N steps.")
    parser.add_argument("--eval_steps", type=int, default=20, help="Evaluate every N steps.")
    parser.add_argument("--eval_strategy", type=str, default="steps", help="Evaluation strategy ('steps' or 'epoch').")
    parser.add_argument("--save_steps", type=int, default=20, help="Save checkpoint every N steps.")
    parser.add_argument("--save_strategy", type=str, default="steps", help="Save strategy ('steps' or 'epoch').")
    parser.add_argument("--output_dir", type=str, default="./output_dpo", help="Output directory for checkpoints.")
    parser.add_argument("--save_merged", action='store_true', help="Save a merged version of the model at the end.")

    # Optimizer and scheduler
    parser.add_argument("--optim", type=str, default="adamw_8bit", help="Optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Learning rate scheduler type.")

    # Other
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of dataloader workers.")

    args = parser.parse_args()
    main(args)

