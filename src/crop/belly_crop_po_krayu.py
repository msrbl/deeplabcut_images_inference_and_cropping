import csv
import os
import cv2
import numpy as np

# Обновлённый порядок точек (индексация с нуля)
ORDERED_INDICES = [0, 1, 2, 5, 8, 11, 14, 17, 20, 19, 18, 15, 12, 9, 6, 3, 0]

def parse_coords_from_row(row):
    coords = []
    for i in range(1, len(row), 3):  # каждая тройка — x, y, likelihood
        try:
            x = float(row[i])
            y = float(row[i + 1])
            coords.append((x, y))
        except (IndexError, ValueError):
            continue
    return coords

def crop_polygon_from_ordered_points(img, all_points, ordered_indices):
    polygon_points = []
    for i in ordered_indices:
        if i < len(all_points):
            polygon_points.append(all_points[i])
    if len(polygon_points) < 3:
        return None, None, None

    contour = np.round(np.array(polygon_points)).astype(np.int32).reshape((-1, 1, 2))

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 255)
    masked = cv2.bitwise_and(img, img, mask=mask)

    x, y, w, h = cv2.boundingRect(contour)
    return masked[y:y+h, x:x+w], polygon_points, contour

def process_csv_and_images(csv_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)

        for idx, row in enumerate(reader):
            if idx < 4:
                continue

            img_path = row[0]
            if not os.path.isfile(img_path):
                print(f"⚠️ Не найдено изображение: {img_path}")
                continue

            coords = parse_coords_from_row(row)
            if len(coords) < 3:
                print(f"⚠️ Недостаточно координат: {img_path}")
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Не удалось открыть изображение: {img_path}")
                continue

            cropped, polygon_points, contour = crop_polygon_from_ordered_points(img, coords, ORDERED_INDICES)
            if cropped is None:
                print(f"❌ Недостаточно валидных точек для обрезки: {img_path}")
                continue

            # Отрисовка всех точек
            for i, (x, y) in enumerate(coords):
                pt = (int(x), int(y))
                color = (0, 255, 0) if i in ORDERED_INDICES else (0, 0, 255)
                cv2.circle(img, pt, 5, color, -1)
                cv2.putText(img, str(i+1), (pt[0]+5, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            # Сохраняем результат
            filename = os.path.basename(img_path)
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, cropped)

            print(f"✅ Обработано: {img_path}")
            print("📐 Контур из точек:")
            print(np.array(polygon_points))
            print(f"💾 Сохранено: {save_path}\n")



# Запуск
process_csv_and_images('D:/crop/bra/annotations.csv', 'D:/crop/save_crop')