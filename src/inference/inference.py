import cv2

from pathlib import Path
from typing import Union
from deeplabcut import analyze_images

from config import logger

def resize_images(
    data_path: Union[str, Path],
    output_path: Union[str, Path],
    image_scale: float = 0.25
    ) -> Path:
    """
    Изменяет размер изображений в указанной папке и её подпапках.

    Args:
        dir (Union[str, Path]): Путь к директории с изображениями.
        image_scale (float): Коэффициент масштабирования изображений (от 0 до 1).

    Returns:
        Path: Путь к папке с изменёнными изображениями.

    Raises:
        ValueError: Если image_scale вне диапазона (0, 1].
        Exception: Если происходит ошибка при изменении размеров изображений.
    """
    data_path = Path(data_path)
    output_path = Path(output_path)
    
    if not 0 < image_scale <= 1:
        raise ValueError("image_scale должен быть в диапазоне (0, 1].")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    try:
        logger.info(f"Поиск изображений в {data_path}")
        images_paths = sorted(p for p in data_path.rglob("*") if p.suffix.lower() in exts)
        logger.debug(f"Найдено {len(images_paths)} изображений.")

        logger.info(f"Создание директории для измененных изображений: {output_path}")
        output_path.mkdir(exist_ok=True)

        for path in images_paths:
            try:
                img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    logger.warning(f"Не удалось прочитать изображение: {path}")
                    continue
                height, width = img.shape[:2]
                resized_height = max(1, int(height * image_scale))
                resized_width = max(1, int(width * image_scale))
                resized_shape = (resized_width, resized_height)
                resized_img = cv2.resize(img, resized_shape)

                new_path = output_path / path.name
                cv2.imwrite(str(new_path), resized_img)
            except Exception as e:
                logger.error(f"Ошибка при обработке изображения {path}: {e}")

        logger.info(f"Изменение размера изображений завершено. Результаты в: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Ошибка при изменении размеров изображений: {e}")
        raise

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
    resize_path = output_path / "resized_images"
    labels_path = output_path / "labels"

    labels_path.mkdir(exist_ok=True)
    resized_images_dir = resize_images(data_path, resize_path)

    analyze_images(
        config_path,
        resized_images_dir,
        frame_type=".JPG",
        device="cpu",
        save_as_csv=True,
        destfolder=labels_path
        )
    
    return labels_path