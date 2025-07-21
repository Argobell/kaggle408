import logging
import argparse
from pathlib import Path
from datasets import load_dataset, DatasetDict
from PIL import Image
from typing import Tuple

# Setup logging to record execution results
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def resize_image_with_aspect_ratio(image: Image.Image, max_size: int) -> Image.Image:
    """
    Resizes an image to a maximum size while maintaining aspect ratio.

    Args:
        image (Image.Image): The PIL Image object.
        max_size (int): The maximum dimension (width or height).

    Returns:
        Image.Image: The resized PIL Image object.
    """
    width, height = image.size
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        return image.resize((new_width, new_height))
    return image

def create_hf_dataset(image_folder: str, json_path: str, max_image_size: int) -> Tuple[object, int, int]:
    """
    Create a Hugging Face dataset from images and a jsonl file.
    Images are resized to the specified dimensions.

    Args:
        image_folder (str): Path to the folder containing images.
        json_path (str): Path to the jsonl file.
        max_image_size (int): The maximum size for the images.

    Returns:
        A tuple containing the dataset object, count of total samples, and failed samples.
    """
    json_path = Path(json_path)
    image_folder = Path(image_folder)

    logging.info(f"Loading JSON data from {json_path}...")
    json_data = load_dataset('json', data_files=str(json_path), split='train')

    logging.info(f"Loading, resizing images to max size {max_image_size}, and adding them to the dataset...")

    def add_and_resize_image(example):
        image_path = image_folder / example['file_name']
        try:
            # Open image, convert to RGB (for consistency), and resize
            image = Image.open(image_path).convert("RGB")
            image = resize_image_with_aspect_ratio(image, max_image_size)
            example['image'] = image
        except FileNotFoundError:
            logging.warning(f"Image file not found: {image_path}")
            example['image'] = None
        return example

    dataset = json_data.map(add_and_resize_image, batched=False)

    total_samples = len(dataset)
    # Filter out samples where the image failed to load
    dataset = dataset.filter(lambda example: example['image'] is not None)
    successful_samples = len(dataset)
    failed_samples = total_samples - successful_samples

    if failed_samples > 0:
        logging.warning(f"Removed {failed_samples} samples with missing images.")

    return dataset, total_samples, failed_samples

def split_and_save_as_parquet(dataset, output_path: str, train_split: float, test_split: float):
    """
    Splits a dataset into train, test, and validation sets,
    and saves them as Parquet files.

    Args:
        dataset (Dataset): The Hugging Face dataset object to split and save.
        output_path (str): Path to the folder to save the Parquet files.
        train_split (float): The proportion of the dataset for training.
        test_split (float): The proportion of the dataset for testing.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory for Parquet files created at: {output_path}")

    # Split the dataset
    train_test_split_size = 1.0 - train_split
    split_dataset = dataset.train_test_split(test_size=train_test_split_size, seed=42)

    # The new test size is relative to the remainder, so we adjust the proportion
    validation_test_split_size = test_split / (1.0 - train_split)
    test_val_split = split_dataset['test'].train_test_split(test_size=validation_test_split_size, seed=42)

    final_splits = DatasetDict({
        'train': split_dataset['train'],
        'test': test_val_split['train'],
        'val': test_val_split['test']
    })

    logging.info(f"Dataset split into: Train ({len(final_splits['train'])}), Test ({len(final_splits['test'])}), Validation ({len(final_splits['val'])}).")

    # Define output filenames
    output_files = {
        "train": "train-00000-of-00001.parquet",
        "test": "test-00000-of-00001.parquet",
        "val": "val-00000-of-00001.parquet"
    }

    # Save each split to a Parquet file
    for split_name, dset in final_splits.items():
        output_file_path = output_path / output_files[split_name]
        logging.info(f"Saving '{split_name}' split to {output_file_path}...")
        dset.to_parquet(output_file_path)

    logging.info("All splits have been successfully saved in Parquet format.")
    return final_splits, output_files

def main(args):
    """
    Main function to orchestrate the dataset creation, splitting, and saving process.
    """
    logging.info(f"Starting dataset creation with image folder: '{args.image_folder}' and json: '{args.json_path}'")

    # Validate split proportions
    if not (0 < args.train_split < 1 and 0 < args.test_split < 1 and 0 < args.val_split < 1):
        raise ValueError("Split proportions must be between 0 and 1.")
    if round(args.train_split + args.test_split + args.val_split, 5) != 1.0:
        raise ValueError("The sum of split proportions must be 1.0.")

    # Step 1: Create the initial dataset from images and JSON
    dataset, total_samples, failed_samples = create_hf_dataset(
        image_folder=args.image_folder,
        json_path=args.json_path,
        max_image_size=args.max_image_size
    )

    # Step 2: Split the dataset and save as Parquet files
    final_splits, output_files = split_and_save_as_parquet(
        dataset=dataset,
        output_path=args.output_path,
        train_split=args.train_split,
        test_split=args.test_split
    )

    # Final summary
    successful_samples = total_samples - failed_samples
    success_rate = (successful_samples / total_samples) * 100 if total_samples > 0 else 0.0

    print("\n--- Execution Summary ---")
    print(f"Total samples processed: {total_samples}")
    print(f"Successful samples (image found): {successful_samples}")
    print(f"Failed samples (image not found): {failed_samples}")
    print(f"Success rate: {success_rate:.2f}%")
    print("-" * 20)
    print(f"Dataset split and saved to: {args.output_path}")
    print(f"Splits created: Train ({len(final_splits['train'])}), Test ({len(final_splits['test'])}), Validation ({len(final_splits['val'])})")
    print("Output files:")
    for split_name in final_splits.keys():
        print(f"  - {output_files[split_name]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and JSON into a split Parquet dataset.")
    # Data source arguments
    parser.add_argument("--image_folder", type=str, default="metadata/MM_Math", help="Path to the folder containing the images.")
    parser.add_argument("--json_path", type=str, default="metadata/MM_Math.jsonl", help="Path to the JSONL data file.")
    parser.add_argument("--max_image_size", type=int, default=512, help="The maximum size (width or height) for the images, preserving aspect ratio.")
    # Output and splitting arguments
    parser.add_argument("--output_path", type=str, default="dataset", help="Path to the folder to save the final Parquet files.")
    parser.add_argument("--train_split", type=float, default=0.9, help="Proportion for the training set.")
    parser.add_argument("--test_split", type=float, default=0.05, help="Proportion for the test set.")
    parser.add_argument("--val_split", type=float, default=0.05, help="Proportion for the validation set.")

    args = parser.parse_args()
    main(args)
