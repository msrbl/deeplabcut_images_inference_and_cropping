import argparse
import shutil

from pathlib import Path

from src.crop.belly_cropper import process_images_crop
from src.inference.inference import process_images
from src.config import APP_DIR, TEMP_DIR, RESULT_DIR

def main():
    if TEMP_DIR.exists() and RESULT_DIR.exists():
        shutil.rmtree(RESULT_DIR)
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    
    inference_path = APP_DIR / "inference"
    
    parser = argparse.ArgumentParser(description="Process images using DeepLabCut.")
    parser.add_argument("--data_path", type=str, help="Path to the data directory.")
    args = parser.parse_args()
    
    data_path = args.data_path if args.data_path else ""
    
    label_dir = process_images(
        parent_path = inference_path,
        data_path = data_path,
        output_path = TEMP_DIR
        )
    
    csv_path = next(label_dir.rglob("*.csv"))
    
    process_images_crop(
        csv_path=csv_path,
        output_dir=RESULT_DIR,
        )
    
if __name__ == "__main__":
    main()