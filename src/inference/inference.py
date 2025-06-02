import cv2

from PIL import Image, ExifTags
from pathlib import Path
from typing import Union
from deeplabcut import analyze_images

from config import logger

def process_images(
    parent_path: Union[str, Path],
    data_path: Union[str, Path],
    output_path: Union[str, Path],
    ) -> Path:
    """
    
    """
    parent_path = Path(parent_path)
    data_path = Path(data_path)
    output_path = Path(output_path)
    
    config_path = parent_path / "DLC-Project-exp-2025-04-28" / "config.yaml"

    output_path.mkdir(exist_ok=True)

    analyze_images(
        config_path,
        data_path,
        frame_type=".JPG",
        device="cpu",
        save_as_csv=True,
        destfolder=output_path
        )
    
    return output_path