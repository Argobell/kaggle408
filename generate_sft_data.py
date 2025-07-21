
import os
import logging
import argparse
from pathlib import Path
from datasets import load_dataset, Dataset
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_system_prompt(prompt_path: str) -> str:
    """Loads the system prompt from a text file."""
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logging.error(f"Prompt file not found at: {prompt_path}")
        raise
    except Exception as e:
        logging.error(f"Error reading prompt file: {e}")
        raise

def clean_solution(client: OpenAI, model: str, solution_text: str, system_prompt: str) -> str:
    """
    Uses OpenAI API to clean and reformat a solution string.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": solution_text}
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"API call failed: {e}")
        return f"Error: API call failed. Details: {e}"

def main(args):
    """
    Main function to load a specific data split, process it, and save the result.
    """
    load_dotenv()
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
    )
    
    system_prompt = load_system_prompt(args.prompt_path)
    
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find the specific parquet file for the given split
    try:
        input_file = next(input_path.glob(f'{args.spilt}-*.parquet'))
    except StopIteration:
        logging.error(f"No parquet file found for split '{args.spilt}' in {input_path}")
        return

    logging.info(f"Loading split '{args.spilt}' from {input_file}...")
    original_dataset = load_dataset('parquet', data_files=str(input_file), split='train')
    
    if args.limit > 0:
        original_dataset = original_dataset.select(range(args.limit))
        logging.info(f"Processing a limited number of records: {args.limit}")

    def update_record(record):
        cleaned_solution = clean_solution(client, args.model, record['solution'], system_prompt)
        return {
            'question': record['question'],
            'solution': cleaned_solution,
            'image': record['image'],
            'difficult': record['difficult']
        }

    processed_records = [update_record(rec) for rec in tqdm(original_dataset, desc=f"Processing {args.spilt} split")]
    
    processed_dataset = Dataset.from_list(processed_records)

    # Save the processed split to a new parquet file
    output_file = output_path / f"{args.spilt}.parquet"
    processed_dataset.to_parquet(output_file)
    logging.info(f"Successfully saved processed split '{args.spilt}' to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and reformat a specific dataset split using an API and save as Parquet.")
    parser.add_argument("--input_path", type=str, default="dataset", help="Path to the input dataset folder.")
    parser.add_argument("--output_path", type=str, default="sft_dataset", help="Path to save the output Parquet files.")
    parser.add_argument("--prompt_path", type=str, default="prompt/clendata.txt", help="Path to the system prompt text file.")
    parser.add_argument("--spilt", type=str, required=True, help="The dataset split to process (e.g., 'train', 'test', 'val').")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="Model to use for cleaning solutions.")
    parser.add_argument("--limit", type=int, default=-1, help="Limit the number of records to process for testing. -1 for no limit.")
    
    args = parser.parse_args()
    main(args)
