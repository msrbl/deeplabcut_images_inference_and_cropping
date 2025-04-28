from pathlib import Path
from typing import List, Union, Optional
import cv2
import pandas as pd
from dlclive import DLCLive, Processor

def _collect_images(folder: Union[str, Path]) -> List[Path]:
    """Return all image paths in *folder* (recursively, common formats)."""
    folder = Path(folder)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in exts)

def infer_folder(
    img_dir: Union[str, Path],
    model_dir: Union[str, Path],
    output_csv: Optional[Union[str, Path]] = None,
    device: str = "cpu",
    show_progress: bool = True,
):
    """Run DeepLabCut inference on all images in *img_dir* using *model_dir*.

    Parameters
    ----------
    img_dir : str|Path
        Directory that contains images. Searches recursively.
    model_dir : str|Path
        Path to a folder with snapshot produced by DeepLabCut (PyTorch).
    output_csv : str|Path, optional
        If given, save a CSV with pose coordinates for each image.
        Columns: *image*, then pairs *(x_i, y_i, p_i)* for every key‑point.
    device : {"cuda", "cpu"}
        Where to run inference.
    show_progress : bool
        If *True*, prints basic progress info every 50 images.
    """

    img_paths = _collect_images(img_dir)
    if not img_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")

    predictor = DLCLive(str(model_dir), processor=Processor(device=device))

    # Pre‑warm with first image to allocate tensors
    first_img = cv2.imread(str(img_paths[0]))
    if first_img is None:
        raise RuntimeError(f"Cannot read {img_paths[0]}")
    predictor.init_inference(first_img.shape)

    data_rows = []
    for idx, img_path in enumerate(img_paths, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Cannot read {img_path}")

        # BGR -> RGB for DLC models
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pose = predictor.get_pose(rgb)  # (n_points, 3) array (x, y, p)

        row = [str(img_path)] + pose.flatten(order="C").tolist()
        data_rows.append(row)

        if show_progress and idx % 50 == 0:
            print(f"Processed {idx}/{len(img_paths)} images…")

    col_headers = ["image"]
    n_points = pose.shape[0]
    for i in range(n_points):
        col_headers += [f"x{i+1}", f"y{i+1}", f"p{i+1}"]

    df = pd.DataFrame(data_rows, columns=col_headers)

    if output_csv:
        output_csv = Path(output_csv)
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

    return df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepLabCut pose inference on a folder of images")
    parser.add_argument("img_dir", type=Path, help="Path to folder with images")
    parser.add_argument("weights", type=Path, help="Path to DLC .pt weight file")
    parser.add_argument("--csv", type=Path, help="Optional path to save CSV with predictions")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")

    args = parser.parse_args()
    infer_folder(args.img_dir, args.weights, args.csv, device=args.device)