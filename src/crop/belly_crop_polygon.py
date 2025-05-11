import csv
import os
import cv2
import numpy as np


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


def crop_polygon_from_points(img, points):
    points = np.array(points, dtype=np.float32)

    # Создаём маску и обрезаем по всем точкам
    contour = np.round(points).astype(np.int32).reshape((-1, 1, 2))

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 255)

    masked = cv2.bitwise_and(img, img, mask=mask)

    x, y, w, h = cv2.boundingRect(contour)
    cropped = masked[y : y + h, x : x + w]

    # Переносим координаты в систему обрезанного изображения
    offset_points = points - np.array([x, y], dtype=np.float32)

    # Отрисовываем точки и подписи
    for idx, (px, py) in enumerate(offset_points):
        cv2.circle(
            cropped, (int(px), int(py)), radius=5, color=(0, 255, 0), thickness=-1
        )
        cv2.putText(
            cropped,
            str(idx),
            (int(px) + 5, int(py) - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return cropped, points, contour


def process_csv_and_images(csv_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)

        for idx, row in enumerate(reader):
            if idx < 4:
                continue  # пропускаем мета-строки

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

            cropped, original_points, contour = crop_polygon_from_points(img, coords)

            print(f"✅ Обрабатывается: {img_path}")
            print("📌 Координаты из CSV:")
            print(np.array(original_points))
            print("📐 Контур для обрезки (после преобразования):")
            print(contour.reshape(-1, 2))  # плоский вид

            # Сохраняем результат
            filename = os.path.basename(img_path)
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, cropped)
            print(f"💾 Сохранено: {save_path}\n")


# Запуск
process_csv_and_images("D:/crop/bra/annotations.csv", "D:/crop/save_crop")
