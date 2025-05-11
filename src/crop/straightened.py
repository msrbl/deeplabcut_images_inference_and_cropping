import csv
import os
import cv2
import numpy as np
from math import atan2, degrees

# Конфигурация
ORDERED_INDICES = [0, 1, 2, 5, 8, 11, 14, 17, 20, 19, 18, 15, 12, 9, 6, 3, 0]
SECTORS = [
    [0, 1, 2, 3, 4, 5],   # Сектор 1
    [3, 4, 5, 6, 7, 8],   # Сектор 2
    [6, 7, 8, 9, 10, 11], # Сектор 3
    [9, 10, 11, 12, 13, 14],
    [12, 13, 14, 15, 16, 17],
    [15, 16, 17, 18, 19, 20]
]
OUTPUT_WIDTH = 172
OUTPUT_HEIGHT = 744
MIN_SPACING = 0

# Цвета границ для каждого сектора
BORDER_COLORS = [
    (0, 0, 255),   # Красный
    (0, 255, 0),   # Зеленый
    (255, 0, 0),   # Синий
    (0, 255, 255), # Желтый
    (255, 0, 255), # Пурпурный
    (255, 255, 0)  # Голубой
]
BORDER_SIZE = 2
TARGET_SCALE = 0.8  # Коэффициент уменьшения

def parse_coords_from_row(row):
    coords = []
    for i in range(1, len(row), 3):
        try:
            x, y = float(row[i]), float(row[i+1])
            coords.append((x, y))
        except (IndexError, ValueError):
            continue
    return coords

def crop_polygon(img, points, indices):
    polygon = [points[i] for i in indices if i < len(points)]
    if len(polygon) < 3:
        return None, None
    
    contour = np.array(polygon, dtype=np.int32)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 255)
    masked = cv2.bitwise_and(img, img, mask=mask)
    x,y,w,h = cv2.boundingRect(contour)
    return masked[y:y+h, x:x+w], (x,y,w,h)

def extract_sector(img, points, indices):
    """Точное извлечение сектора с минимальной высотой"""
    sector_points = [points[i] for i in indices if i < len(points)]
    if len(sector_points) < 3:
        return None, None
    
    # Получаем выпуклую оболочку точек сектора
    hull = cv2.convexHull(np.array(sector_points, dtype=np.float32))
    
    # Находим ограничивающий прямоугольник
    x, y, w, h = cv2.boundingRect(hull)
    
    # Создаем маску только для текущего сектора
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [hull.astype(np.int32)], 255)
    
    # Применяем маску и обрезаем до точных границ
    sector = cv2.bitwise_and(img, img, mask=mask)
    cropped = sector[y:y+h, x:x+w]
    
    # Рассчитываем новые координаты точек относительно обрезанного изображения
    shifted_points = [(p[0]-x, p[1]-y) for p in sector_points]
    
    return cropped, shifted_points

def align_sector(sector_img, points):
    """Выравнивание сектора с минимальной обрезанной областью"""
    if sector_img is None or len(points) < 6:
        return None
    
    # Находим крайние точки для определения угла наклона
    left_points = [points[0], points[3]]
    right_points = [points[2], points[5]]
    
    # Рассчитываем угол наклона
    angles = []
    for lp, rp in zip(left_points, right_points):
        delta_x = rp[0] - lp[0]
        delta_y = rp[1] - lp[1]
        if delta_x != 0:  # Избегаем деления на ноль
            angle = degrees(atan2(delta_y, delta_x))
            angles.append(angle)
    
    if not angles:
        return None
    
    avg_angle = np.mean(angles)
    
    # Получаем размеры изображения
    h, w = sector_img.shape[:2]
    
    # Вычисляем матрицу поворота
    rotation_matrix = cv2.getRotationMatrix2D((w//2, h//2), avg_angle, 1.0)
    
    # Поворачиваем изображение
    try:
        rotated = cv2.warpAffine(
            sector_img,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        
        # Находим новые границы после поворота
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            return rotated[y:y+h, x:x+w]
        return rotated
        
    except Exception as e:
        print(f"Ошибка при выравнивании: {str(e)}")
        return None

def resize_sector(sector, scale):
    """Уменьшает сектор с сохранением пропорций"""
    if sector is None:
        return None
    h, w = sector.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(sector, (new_w, new_h), interpolation=cv2.INTER_AREA)

def add_border(sector, color, thickness):
    """Добавляет цветную рамку к сектору"""
    if sector is None:
        return None
    return cv2.copyMakeBorder(sector, 
                            thickness, thickness, thickness, thickness,
                            cv2.BORDER_CONSTANT, value=color)

def pack_sectors(sectors):
    """Упаковка секторов с минимальными расстояниями"""
    if len(sectors) != 6:
        return None
    
    # Фильтруем только валидные сектора
    valid_sectors = [s for s in sectors if s is not None and s.size > 0]
    if not valid_sectors:
        return None
    
    # Находим максимальную ширину среди секторов
    max_width = max(s.shape[1] for s in valid_sectors)
    
    # Рассчитываем общую высоту
    total_height = sum(s.shape[0] for s in valid_sectors) + (len(valid_sectors) - 1) * MIN_SPACING
    
    # Создаем результирующее изображение
    result = np.zeros((total_height, max_width, 3), dtype=np.uint8)
    y_offset = 0
    
    # Размещаем сектора с минимальными отступами
    for i, sector in enumerate(valid_sectors):
        h, w = sector.shape[:2]
        x_offset = (max_width - w) // 2
        
        # Вставляем сектор
        result[y_offset:y_offset+h, x_offset:x_offset+w] = sector
        y_offset += h
        
        # Добавляем отступ (кроме последнего сектора)
        if i < len(valid_sectors) - 1:
            result[y_offset:y_offset+MIN_SPACING, :] = [0, 0, 255]  # Красная разделительная линия
            y_offset += MIN_SPACING
    
    return result

def resize_to_output(img):
    """Приведение к финальному размеру"""
    if img is None:
        return np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
    
    h,w = img.shape[:2]
    scale = min(OUTPUT_WIDTH/w, OUTPUT_HEIGHT/h)
    new_w, new_h = int(w*scale), int(h*scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    result = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
    x_offset = (OUTPUT_WIDTH - new_w) // 2
    y_offset = (OUTPUT_HEIGHT - new_h) // 2
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return result

def process_image(img_path, coords, output_dir):
    """Обработка изображения с исправленными проверками"""
    try:
        print(f"\nОбработка: {img_path}")
        
        # 1. Загрузка изображения с явной проверкой
        img = cv2.imread(img_path)
        if img is None or not isinstance(img, np.ndarray):
            print("⚠️ Ошибка загрузки изображения")
            return False
        
        # 2. Обрезка по полигону с проверкой результата
        cropped_result = crop_polygon(img, coords, ORDERED_INDICES)
        if cropped_result[0] is None or not isinstance(cropped_result[0], np.ndarray):
            print("⚠️ Не удалось обрезать по полигону")
            return False
        cropped, crop_rect = cropped_result
        
        # 3. Корректировка координат
        adjusted_coords = [(x-crop_rect[0], y-crop_rect[1]) for (x,y) in coords]
        
        # 4. Извлечение и выравнивание секторов
        aligned_sectors = []
        for i, sector_indices in enumerate(SECTORS, 1):
            try:
                # Явная проверка каждого этапа
                sector_result = extract_sector(cropped, adjusted_coords, sector_indices)
                if sector_result[0] is None or not isinstance(sector_result[0], np.ndarray):
                    print(f"⚠️ Сектор {i}: не удалось извлечь")
                    aligned_sectors.append(None)
                    continue
                
                sector_img, points = sector_result
                aligned = align_sector(sector_img, points)
                if aligned is None or not isinstance(aligned, np.ndarray):
                    print(f"⚠️ Сектор {i}: не удалось выровнять")
                    aligned_sectors.append(None)
                    continue
                
                print(f"✅ Сектор {i}: {aligned.shape[1]}x{aligned.shape[0]}")
                aligned_sectors.append(aligned)
                
            except Exception as e:
                print(f"⚠️ Ошибка в секторе {i}: {str(e)}")
                aligned_sectors.append(None)
                continue
        
        # 5. Упаковка секторов с проверкой
        packed = pack_sectors(aligned_sectors)
        if packed is None or not isinstance(packed, np.ndarray):
            print("⚠️ Не удалось упаковать сектора")
            return False
        
        # 6. Финальное масштабирование
        final_image = resize_to_output(packed)
        if final_image is None or not isinstance(final_image, np.ndarray):
            print("⚠️ Ошибка финального масштабирования")
            return False
        
        # 7. Сохранение результата
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, final_image)
        print(f"✅ Успешно сохранено: {output_path}")
        return True
        
    except Exception as e:
        print(f"⛔ Критическая ошибка: {str(e)}")
        return False

def process_csv(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i < 4:
                continue
                
            img_path = row[0]
            coords = parse_coords_from_row(row)
            if len(coords) >= 21:
                process_image(img_path, coords, output_dir)

if __name__ == "__main__":
    process_csv("D:/crop/bra/annotations.csv", "D:/crop/save_aligned")