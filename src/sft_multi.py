from unsloth import FastVisionModel
import torch
import os
from pathlib import Path
from datasets import load_dataset
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import swanlab
import argparse
from swanlab.integration.transformers import SwanLabCallback

def main(args):

    swanlab_callback = SwanLabCallback(
        project="kaggle408",
        experiment_name="gemma3n-mutlti-finetune",
    )

    # Load model and processor
    model, processor = FastVisionModel.from_pretrained(
        model_name=args.model_name_or_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        attn_implementation="eager",
        use_gradient_checkpointing = "unsloth",
    )

    # Get PEFT model
    model = FastVisionModel.get_peft_model(
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
    def convert2conversation(sample):
        instruction = sample[f"{args.instruction_name}"]
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": sample["image"]},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": sample[f"{args.answer_name}"]}]},
        ]
        return {"messages": conversation}
    
    train_dataset = load_dataset(args.dataset_name, split="train")
    converted_train_dataset = [convert2conversation(sample) for sample in train_dataset]

    if args.do_eval:
        val_dataset = load_dataset(args.dataset_name, split="val")
        converted_val_dataset = [convert2conversation(sample) for sample in val_dataset]
    else:
        converted_val_dataset = None
    
    FastVisionModel.for_training(model) # Enable for training!


    trainer = SFTTrainer(
        model=model,
        train_dataset=converted_train_dataset,
        eval_dataset=converted_val_dataset,
        processing_class=processor.tokenizer,
        data_collator=UnslothVisionDataCollator(model, processor),
        callbacks=[swanlab_callback],
        args=SFTConfig(
            torch_compile = False,
            per_device_train_batch_size=args.per_device_train_batch_size, # Batch size for training
            per_device_eval_batch_size=args.per_device_eval_batch_size, # Batch size for evaluation
            gradient_accumulation_steps=args.gradient_accumulation_steps, # Steps to accumulate gradients
            gradient_checkpointing=args.gradient_checkpointing, # Enable gradient checkpointing for memory efficiency

            # use reentrant checkpointing
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_grad_norm=args.max_grad_norm,  # Maximum norm for gradient clipping
            warmup_ratio=args.warmup_ratio,  # Ratio of total steps for warmup
            num_train_epochs=args.num_train_epochs,  # Number of training epochs
            learning_rate=args.learning_rate, # Learning rate for training
            logging_steps=args.logging_steps, # Steps interval for logging
            eval_steps=args.eval_steps, # Steps interval for evaluation
            eval_strategy=args.eval_strategy,  # Strategy for evaluation
            save_steps=args.save_steps, # Steps interval for saving
            save_strategy=args.save_strategy, # Strategy for saving the model
            metric_for_best_model="eval_loss", # Metric to evaluate the best model
            load_best_model_at_end=True, # Load the best model after training
            optim=args.optim, # Optimizer type
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type, # Type of learning rate scheduler
            seed=3407,
            output_dir=args.output_dir, 
            # You MUST put the below items for vision finetuning:
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=args.max_seq_length,
        )
    )

    trainer_stats = trainer.train()

    torch.cuda.empty_cache()

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    if args.save_merged:
        parent_dir = os.path.dirname(args.output_dir)
        new_subdir = os.path.join(parent_dir, "merged_e4b")
        os.makedirs(new_subdir, exist_ok=True)
        model.save_pretrained_merged(new_subdir,processor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a vision model with Unsloth")
    
    # Model and PEFT parameters
    parser.add_argument("--model_name_or_path", type=str, default="unsloth/phi-3-mini-4k-instruct-vision", help="Model name or path.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--load_in_4bit", action='store_true', help="Load model in 4-bit precision.")
    parser.add_argument("--tune_vision", action='store_true', help="Fine-tune vision layers.")
    parser.add_argument("--tune_language_layers", action='store_true', help="Fine-tune language layers.")
    parser.add_argument("--tune_attention_modules", action='store_true', help="Fine-tune attention modules.")
    parser.add_argument("--tune_mlp_modules", action='store_true', help="Fine-tune MLP modules.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--r", type=int, default=16, help="LoRA r parameter.")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha parameter.")
    
    # Dataset parameters
    parser.add_argument("--instruction_name", type=str, required=True, help="The column name for instructions in the dataset.")
    parser.add_argument("--answer_name", type=str, required=True, help="The column name for answers in the dataset.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset on Hugging Face Hub.")
    
    # Training parameters
    parser.add_argument("--do_eval", action='store_true', help="Perform evaluation during training.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Training batch size per device.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Evaluation batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--gradient_checkpointing", action='store_true', help="Enable gradient checkpointing.")
    parser.add_argument("--max_grad_norm", type=float, default=0.3, help="Maximum gradient norm for clipping.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    
    # Logging and saving
    parser.add_argument("--logging_steps", type=int, default=5, help="Log every N steps.")
    parser.add_argument("--eval_steps", type=int, default=20, help="Evaluate every N steps.")
    parser.add_argument("--eval_strategy", type=str, default="steps", help="Evaluation strategy ('steps' or 'epoch').")
    parser.add_argument("--save_steps", type=int, default=20, help="Save checkpoint every N steps.")
    parser.add_argument("--save_strategy", type=str, default="steps", help="Save strategy ('steps' or 'epoch').")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for checkpoints.")
    parser.add_argument("--save_merged", action='store_true', help="Save a merged version of the model at the end.")
    
    # Optimizer and scheduler
    parser.add_argument("--optim", type=str, default="adamw_8bit", help="Optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Learning rate scheduler type.")
    
    args = parser.parse_args()
    main(args)