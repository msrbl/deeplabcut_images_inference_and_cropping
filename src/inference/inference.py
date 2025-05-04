from pathlib import Path
from typing import List, Union
import cv2

import deeplabcut

def resize_images(folder: Union[str, Path]) -> List[Path]:
    """
    """
    folder = Path(folder)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    original_paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in exts)
    
    resized_dir = folder.parent / "resized_images"
    resized_dir.mkdir(exist_ok=True)
    
    for image_path in original_paths:
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        height, width = img.shape[:2]
        resized_height = max(1, height // 4)
        resized_width = max(1, width // 4)
        resized_shape = (resized_width, resized_height)
        resized_img = cv2.resize(img, resized_shape)
        
        new_path = resized_dir / image_path.name
        cv2.imwrite(str(new_path), resized_img)
    
    return resized_dir

if __name__ == "__main__":
    parent_dir = Path(__file__).parent
    config_path = parent_dir / "DLC-Project-exp-2025-04-28" / "config.yaml"
    image_dir = parent_dir / "images"
    dest_dir = parent_dir / "results"

    resized_images = resize_images(image_dir)

    deeplabcut.analyze_images(config_path, resized_images, frame_type=".JPG", device="cpu", save_as_csv=True, destfolder=dest_dir)