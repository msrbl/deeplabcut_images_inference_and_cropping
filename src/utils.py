import cv2

from pathlib import Path
from PIL import Image, ExifTags
from typing import Union

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
                image = Image.open(path)
                exif = image._getexif()
                if exif is not None:
                    orientation_key = next((key for key, val in ExifTags.TAGS.items() if val == 'Orientation'), None)
                    if orientation_key is not None and orientation_key in exif:
                        orientation = exif[orientation_key]
                        if orientation == 3:
                            image = image.rotate(180, expand=True)
                        elif orientation == 6:
                            image = image.rotate(270, expand=True)
                        elif orientation == 8:
                            image = image.rotate(90, expand=True)
                image.save(path)
                
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

def group_images_by_id(result_dir):
    for img_file in Path(result_dir).glob("*.*"):
        if img_file.is_file():
            filename = img_file.stem
            if len(filename) >= 4 and filename[-4:].isdigit():
                group = filename[-4:]
                group_folder = Path(result_dir) / group
                group_folder.mkdir(exist_ok=True)
                img_file.rename(group_folder / img_file.name)