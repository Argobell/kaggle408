import os
import logging
import argparse
import random
from pathlib import Path
import torch
from datasets import load_dataset, Dataset
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from tqdm import tqdm

torch.set_float32_matmul_precision('high')
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_rejection_prompts(prompt_dir: str):
    """Loads rejection prompts from .txt files in a directory."""
    prompts = {}
    logging.info(f"Loading rejection prompts from '{prompt_dir}'...")
    rejection_filenames = ["calculation_error.txt", "logical_fallacy.txt", "process_error.txt"]

    try:
        for filename in rejection_filenames:
            strategy_name = filename.removesuffix(".txt")
            file_path = os.path.join(prompt_dir, filename)

            if not os.path.exists(file_path):
                logging.warning(f"Prompt file '{filename}' not found. Skipping.")
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                prompts[strategy_name] = f.read()

            logging.info(f"  - Loaded prompt: {strategy_name}")
    except Exception as e:
        logging.error(f"Failed to load prompts: {e}")
        raise
    if not prompts:
        raise FileNotFoundError("No valid prompt files were found.")
    return prompts

def generate_rejected_response(model, processor, example, rejection_prompts):

    try:
        strategy_name, system_prompt_text = random.choice(list(rejection_prompts.items()))

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt_text}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example["image"]},
                    {"type": "text", "text": example["question"]}
                ]
            }
        ]
        
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = model.generate(
                **inputs, max_new_tokens=1024, do_sample=True,
                temperature=1.0, top_p=0.95,
            )
            generation = generation[0][input_len:]

        rejected_text = processor.decode(generation, skip_special_tokens=True).strip()

    except Exception as e:
        logging.error(f"Failed to process a record due to: {e}. The 'rejected' field will contain the error.")
        rejected_text = f"GENERATION_ERROR: {e}"
    
    return rejected_text

def main(args):

    logging.info("--- DPO Dataset Generation Script Started (Loop-based) ---")

    prompt_dir = Path(args.prompt_dir)
    model_path = Path(args.model_path)
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rejection_prompts = load_rejection_prompts(prompt_dir)

    logging.info(f"Loading model from '{model_path}'...")

    model = Gemma3nForConditionalGeneration.from_pretrained(
        model_path, 
        device_map="cuda:0", 
        torch_dtype=torch.bfloat16,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path)

    logging.info(f"Loading split '{args.split}' from {dataset_path}...")

    dataset = load_dataset(str(dataset_path), split=args.split)
    logging.info(f"Loaded {len(dataset)} examples for split '{args.split}'.")

    if args.limit > 0:
        dataset = dataset.select(range(args.limit))
        logging.info(f"Processing a limited subset of {args.limit} records.")
    
    def update_record(example):
        rejected_text = generate_rejected_response(model, processor, example, rejection_prompts)
        return {
            'image': example['image'],
            'question': example['question'],
            'chosen': example['solution'],
            'rejected': rejected_text
        }
    
    processed_records = [update_record(example) for example in tqdm(dataset, desc=f"Processing {args.split} split")]

    dpo_dataset = Dataset.from_list(processed_records)

    logging.info(f"Shuffling '{args.split}' split with seed {args.seed}...")
    dpo_dataset = dpo_dataset.shuffle(seed=args.seed)

    logging.info(f"Final schema for split '{args.split}': {dpo_dataset.features}")
    output_file = output_dir / f"{args.split}.parquet"
    dpo_dataset.to_parquet(output_file)
    logging.info(f"Successfully saved processed split '{args.split}' to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt_dir", type=str, default="/root/autodl-tmp/kaggle408/prompt", help="Directory containing rejection prompt files")
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/model/gemma3n_E2B", help="Path to the model")
    parser.add_argument("--dataset_path", type=str, default="/root/autodl-tmp/kaggle408/dataset/gek408", help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/kaggle408/dataset/gek408-dpo", help="Directory to save output files")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split to process")
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of records to process (-1 for all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")

    
    args = parser.parse_args()
    main(args)
