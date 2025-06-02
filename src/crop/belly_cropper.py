import csv
import os
import cv2
import numpy as np

from config import ORDERED_INDICES, TARGET_WIDTH, TARGET_HEIGHT, logger

def parse_coords_from_row(row):
    """
    Извлекает координаты x, y из строки CSV.

    Args:
        row (list): Строка из CSV файла.

    Returns:
        list: Список кортежей (x, y) координат.
    """
    coords = []
    for i in range(1, len(row), 3):  
        try:
            x = float(row[i])
            y = float(row[i + 1])
            coords.append((x, y))
        except (IndexError, ValueError) as e:
            logger.error(f"Ошибка при парсинге координат: {e}")
            continue
    return coords

def crop_polygon_from_ordered_points(img, all_points, ordered_indices, target_width, target_height):
    """
    Обрезает полигон из изображения на основе упорядоченного списка точек и изменяет размер.

    Args:
        img (numpy.ndarray): Изображение для обрезки.
        all_points (list): Список всех точек координат.
        ordered_indices (list): Список индексов, определяющих порядок точек для полигона.
        target_width (int): Целевая ширина изображения.
        target_height (int): Целевая высота изображения.

    Returns:
        tuple: Кортеж, содержащий обрезанное изображение, список точек полигона и контур.
    """
    polygon_points = []
    for i in ordered_indices:
        if i < len(all_points):
            polygon_points.append(all_points[i])
    if len(polygon_points) < 3:
        logger.warning("Недостаточно точек для обрезки полигона.")
        return None, None, None

    contour = np.round(np.array(polygon_points)).astype(np.int32).reshape((-1, 1, 2))

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 255)
    masked = cv2.bitwise_and(img, img, mask=mask)

    x, y, w, h = cv2.boundingRect(contour)
    cropped = masked[y:y+h, x:x+w]

    # Изменение размера обрезанного изображения
    cropped = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_AREA)

    return cropped, polygon_points, contour

def process_images_crop(csv_path, output_dir, ordered_indices=ORDERED_INDICES, target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT):
    """
    Обрабатывает изображения на основе данных из CSV файла, обрезает их и сохраняет.

    Args:
        csv_path (str): Путь к CSV файлу с данными.
        output_dir (str): Путь к директории для сохранения обрезанных изображений.
        ordered_indices (list): Список индексов, определяющих порядок точек для полигона.
        target_width (int): Целевая ширина изображения.
        target_height (int): Целевая высота изображения.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        with open(csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile)

            for idx, row in enumerate(reader):
                if idx < 4:
                    continue

                img_path = row[0]
                if not os.path.isfile(img_path):
                    logger.warning(f"Изображение не найдено: {img_path}")
                    continue

                coords = parse_coords_from_row(row)
                if len(coords) < 3:
                    logger.warning(f"Недостаточно координат для изображения: {img_path}")
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    logger.error(f"Не удалось открыть изображение: {img_path}")
                    continue

                cropped, polygon_points, contour = crop_polygon_from_ordered_points(img, coords, ordered_indices, target_width, target_height)
                if cropped is None:
                    logger.warning(f"Недостаточно валидных точек для обрезки: {img_path}")
                    continue
                
                for i, (x, y) in enumerate(coords):
                    pt = (int(x), int(y))
                    color = (0, 255, 0) if i in ordered_indices else (0, 0, 255)
                    cv2.circle(img, pt, 5, color, -1)
                    cv2.putText(img, str(i+1), (pt[0]+5, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                
                filename = os.path.basename(img_path)
                save_path = os.path.join(output_dir, filename)
                cv2.imwrite(save_path, cropped)

            logger.info(f"Обрезка изображений завершена. Результаты сохранены в: {output_dir}")
    except Exception as e:
        logger.error(f"Произошла ошибка при обработке: {e}")