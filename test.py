from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# 1. Укажите путь к скачанному файлу best.pt
model_path = r"D:\test znakov\models\best.pt" 
model = YOLO(model_path)

# 2. Укажите путь к картинке, которую хотите проверить
# (Возьмите любую картинку со знаками из вашей папки)
image_path = r"D:\test znakov\10984061.jpg"

# 3. Запускаем распознавание
# conf=0.25 - стандартный порог для реальной работы (отсекает явный мусор)
results = model(image_path, conf=0.25, imgsz=640)

# 4. Показываем результат
for r in results:
    # Рисуем рамки
    im_array = r.plot()
    
    # Конвертируем цвета для отображения в matplotlib (OpenCV использует BGR, а matplotlib - RGB)
    im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
    
    # Показываем картинку
    plt.figure(figsize=(12, 8))
    plt.imshow(im_rgb)
    plt.axis('off')
    plt.show()
    
    # Выводим в консоль, что нашла модель
    if len(r.boxes) == 0:
        print("Модель ничего не нашла на этой картинке.")
    else:
        print(f"Найдено объектов: {len(r.boxes)}")
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]
            print(f" - Знак: {class_name} (Уверенность: {conf:.2f})")