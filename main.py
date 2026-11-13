from ultralytics import YOLO

# Загружаем модель
model = YOLO("yolov8n.pt")

# Начинаем обучение
results = model.train(
    data="data.yaml",
    epochs=2,             # Всего 2 эпохи для теста
    imgsz=320,            # Уменьшили размер до 320 для скорости (на CPU это критично)
    batch=8,              # Маленький батч
    device="cpu",         # Используем процессор
    workers=2,            
    fraction=0.1,         # ВАЖНО: берем только 10% картинок из датасета (около 4000 шт)
    name="test_run"
)