import os
import json
import base64
import logging
import argparse
import random
import asyncio
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Set

import openai
import pandas as pd
from PIL import Image
from datasets import load_dataset
from tqdm.asyncio import tqdm as async_tqdm
from dotenv import load_dotenv

load_dotenv()

# --- Constants and Configuration ---
GENERATOR_MODEL_NAME = "deepseek-chat"
JUDGE_MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"
HIGH_QUALITY_PROBABILITY = 0.7
JUDGE_SCORE_WEIGHTS = {
    "correctness": 0.4,
    "logic": 0.3,
    "conciseness": 0.2,
    "persona_adherence": 0.1
}

# --- Prompts ---

GENERATOR_SYSTEM_PROMPT_HIGH_QUALITY = """
# ROLE: AI Problem-Solving Assistant

## PERSONA
You are an enthusiastic and friendly study buddy. Your goal is to help users understand *how* to solve a problem, not just give them the final answer. Your tone should be encouraging, patient, and clear. Imagine you are explaining this to a friend you are studying with.

## TASK
Based on the provided reference solution, you must generate a new, detailed, step-by-step explanation for how to solve the problem.

## INSTRUCTIONS
1.  **Adopt the Persona**: Maintain a warm and encouraging tone.
2.  **Logical Breakdown**: Deconstruct the logic into simple, easy-to-follow steps.
3.  **Clarity and Conciseness**: Keep your explanation concise but complete.
4.  **Input**: You will be given a reference solution.
5.  **Output**: Provide only the newly generated explanation.
"""

GENERATOR_SYSTEM_PROMPT_WITH_ERROR = """
# ROLE: AI Problem-Solving Assistant (with a deliberate flaw)

## PERSONA
You are an enthusiastic and friendly study buddy. Your goal is to help users, but you sometimes make common mistakes. Your tone MUST remain encouraging and confident, even though your solution contains an error.

## TASK
Based on the provided reference solution, generate a new, detailed, step-by-step explanation that **intentionally includes one common, believable error**.

## INSTRUCTIONS
1.  **Adopt the Persona**: Maintain a warm, encouraging, and confident tone. The explanation should sound helpful, masking the error.
2.  **Introduce a Subtle Flaw**: Your explanation MUST contain one of the following common errors:
    *   A minor calculation mistake (e.g., 2*3=5).
    *   Misinterpreting a part of the question slightly.
    *   Applying a correct formula incorrectly.
    *   A small logical leap that misses a key step.
3.  **Be Believable**: The error should be subtle and something a real student might do. Do not make the answer nonsensical. The rest of the logic should appear sound.
4.  **Input**: You will be given a reference solution.
5.  **Output**: Provide only the newly generated, flawed explanation.
"""

JUDGE_SYSTEM_PROMPT = """
# ROLE: Impartial Judge for Educational Content

## TASK
You are an expert mathematics teacher. Your task is to evaluate two student-generated answers ("Answer A" and "Answer B") for a given math problem. You will be provided with the original **question**, the accompanying **image**, and the **official solution** for your reference. Your evaluation must be based on the criteria below, ensuring the answers align with all provided materials.

Even if both answers contain errors, please choose the one that is closer to the correct solution based on the official solution. Prioritize answers with fewer or less severe mistakes. Focus on the following:
- Logical consistency and alignment with the question requirements.
- Severity of any errors (e.g., logic errors > formula misuse > calculation mistakes).
- Whether the error is masked well or clearly misleading.

## EVALUATION CRITERIA
1.  **Correctness (0-5)**: How factually correct is the answer? Does it align with the question, image, and official solution? (0 for completely wrong, 5 for perfect).
2.  **Logic (0-5)**: Is the problem-solving logic sound, clear, and easy to follow? (0 for illogical, 5 for perfectly structured logic).
3.  **Conciseness (0-5)**: Is the explanation straight to the point, avoiding unnecessary text while remaining complete? (0 for overly verbose, 5 for perfectly concise).
4.  **Persona Adherence (0-5)**: How well does the answer embody the persona of a friendly, enthusiastic study buddy? (0 for a cold, robotic tone; 5 for a perfect, engaging persona).

## OUTPUT FORMAT
You MUST respond with a single JSON object. The object should contain two main keys, "scores_a" and "scores_b", which hold the scores for each answer. Include a "reasoning" key to briefly justify your scores.
"""

# --- Caching ---
CACHE = {}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def pil_to_base64_uri(img: Image.Image, format: str = 'PNG') -> str:
    """Converts a PIL Image to a base64 data URI for API calls."""
    buf = BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f'data:image/{format.lower()};base64,{b64}'

def get_async_api_client(api_key: str, base_url: str = None) -> openai.AsyncOpenAI:
    """Retrieves API key from environment and creates an AsyncOpenAI client."""
    return openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

async def async_generate_one_answer(client: openai.AsyncOpenAI, solution: str) -> str:
    """Generates a single answer asynchronously."""
    is_high_quality = random.random() < HIGH_QUALITY_PROBABILITY
    system_prompt = GENERATOR_SYSTEM_PROMPT_HIGH_QUALITY if is_high_quality else GENERATOR_SYSTEM_PROMPT_WITH_ERROR
    
    cache_key = ("generate", solution, system_prompt)
    if cache_key in CACHE:
        return CACHE[cache_key]

    try:
        response = await client.chat.completions.create(
            model=GENERATOR_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the reference solution:\n\n{solution}"},
            ],
            temperature=1,
            stream=False,
            max_tokens=512,
        )
        result = response.choices[0].message.content
        CACHE[cache_key] = result
        return result
    except Exception as e:
        logging.error(f"Error during candidate generation: {e}")
        return None

async def async_judge_answers(
    client: openai.AsyncOpenAI, question: str, solution: str, image: Image.Image, answer_a: str, answer_b: str
) -> Dict[str, Any]:
    """Gets a detailed evaluation asynchronously."""
    image_uri = pil_to_base64_uri(image)
    
    cache_key = ("judge", question, solution, answer_a, answer_b)
    if cache_key in CACHE:
        return CACHE[cache_key]

    user_prompt = f"""**Math Problem:**\n{question}\n\n**Official Solution (for your reference):**\n{solution}\n\n---\n\nNow, please evaluate the following two answers based on all the information provided.\n\n**Answer A:**\n{answer_a}\n\n**Answer B:**\n{answer_b}\n\nProvide your evaluation as a JSON object.\n"""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_uri}}
            ]
        }
    ]
    
    try:
        response = await client.chat.completions.create(
            model=JUDGE_MODEL_NAME,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
            stream=False,
        )
        eval_result = json.loads(response.choices[0].message.content)
        CACHE[cache_key] = eval_result
        return eval_result
    except Exception as e:
        logging.error(f"Error evaluating answers with judge model: {e}")
        return None

def calculate_weighted_score(scores: Dict[str, float]) -> float:
    """Calculates the weighted score."""
    total_score = 0
    for key, weight in JUDGE_SCORE_WEIGHTS.items():
        total_score += scores.get(key, 0) * weight
    return total_score

async def process_item(item: Dict, generator_client: openai.AsyncOpenAI, judge_client: openai.AsyncOpenAI) -> Dict:
    """Processes a single item, returning a DPO record with image filename."""
    question, solution, image = item.get("question"), item.get("solution"), item.get("image")

    if not all([question, solution, image]):
        logging.warning("Skipping item due to missing data.")
        return None

    filename = item.get("file_name")
    if not filename:
        logging.warning(f"Could not determine filename for question: {question[:50]}... Skipping.")
        return None

    answer_a, answer_b = await asyncio.gather(
        async_generate_one_answer(generator_client, solution),
        async_generate_one_answer(generator_client, solution)
    )

    if not all([answer_a, answer_b]) or answer_a == answer_b:
        logging.warning("Could not generate two distinct valid answers. Skipping.")
        return None

    evaluation = await async_judge_answers(judge_client, question, solution, image, answer_a, answer_b)
    if not evaluation or "scores_a" not in evaluation or "scores_b" not in evaluation:
        logging.warning("Could not get a valid evaluation from the judge. Skipping.")
        return None
        
    weighted_score_a = calculate_weighted_score(evaluation["scores_a"])
    weighted_score_b = calculate_weighted_score(evaluation["scores_b"])

    if weighted_score_a == weighted_score_b:
        logging.info("Weighted scores are identical. Skipping.")
        return None
        
    chosen, rejected = (answer_a, answer_b) if weighted_score_a > weighted_score_b else (answer_b, answer_a)

    return {
        "question": question,
        "chosen": chosen,
        "rejected": rejected,
        "filename": os.path.basename(filename),
    }

async def main(args):
    """Main async function to orchestrate the DPO data generation pipeline."""
    logging.info("Starting DPO data generation pipeline...")
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        generator_client = get_async_api_client(api_key = os.getenv("DEEPSEEK_API_KEY"),base_url = os.getenv("DEEPSEEK_BASE_URL"))
        judge_client = get_async_api_client(api_key = os.getenv("MODELSCOPE_API_KEY"), base_url = os.getenv("MODELSCOPE_BASE_URL"))
        logging.info("Async API clients initialized successfully.")
    except ValueError as e:
        logging.error(e)
        return

    # --- State Management: Load existing data to avoid reprocessing ---
    processed_questions: Set[str] = set()
    write_header = True
    if output_path.exists() and os.path.getsize(output_path) > 0:
        try:
            df_existing = pd.read_csv(output_path)
            if 'question' in df_existing.columns:
                processed_questions = set(df_existing['question'])
                write_header = False
                logging.info(f"Found {len(processed_questions)} already processed questions in {output_path}.")
        except Exception as e:
            logging.warning(f"Could not read existing CSV file at {output_path}. Starting fresh. Error: {e}")

    try:
        dataset = load_dataset(args.dataset_path, split="train")
        logging.info(f"Full dataset loaded with {len(dataset)} samples.")
    except Exception as e:
        logging.error(f"Failed to load dataset from '{args.dataset_path}': {e}")
        return

    # --- Filter out processed questions ---
    unprocessed_dataset = dataset.filter(lambda example: example['question'] not in processed_questions)
    logging.info(f"{len(unprocessed_dataset)} unprocessed samples remaining.")

    if len(unprocessed_dataset) == 0:
        logging.info("No new samples to process. Exiting.")
        return

    # --- Select the next batch of samples to process ---
    num_to_process = min(args.max_samples, len(unprocessed_dataset))
    samples_to_process = unprocessed_dataset.select(range(num_to_process))
    logging.info(f"Processing the next {len(samples_to_process)} samples in this run.")

    tasks = [process_item(item, generator_client, judge_client) for item in samples_to_process]
    
    dpo_records = []
    for future in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating DPO Data"):
        result = await future
        if result:
            dpo_records.append(result)

    if not dpo_records:
        logging.warning("No new DPO records were generated in this run.")
        return

    logging.info(f"Generated {len(dpo_records)} new DPO records. Appending to CSV file...")
    df = pd.DataFrame(dpo_records)
    try:
        # Append to CSV without writing header if file exists
        df.to_csv(output_path, mode='a', header=write_header, index=False)
        logging.info(f"DPO data generation complete for this run. Output saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save CSV file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and append a DPO dataset using generator and judge models.")
    
    parser.add_argument("--dataset_path", type=str, default="dataset", help="Path to the Hugging Face dataset directory.")
    parser.add_argument("--output_file", type=str, default="dpo_dataset.csv", help="Path to save the final DPO dataset.")
    parser.add_argument("--max_samples", type=int, default=10, help="Maximum number of new samples to process in one run.")
    
    args = parser.parse_args()
    
    asyncio.run(main(args))