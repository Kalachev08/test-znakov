import os
import cv2
import time
import threading
from ultralytics import YOLO

# --- НАСТРОЙКИ ---
RTSP_URL = "rtsp://192.168.1.254:554/"  # Ваш URL
MODEL_PATH = r"D:\test znakov\models\best.pt"
CONFIDENCE_THRESHOLD = 0.25
IMG_SIZE = 640

# Отключаем спам от FFmpeg в консоли (те самые ошибки bytestream)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|"          # Используем TCP для стабильности
    "fflags;nobuffer|"             # Без лишнего буфера
    "flags;low_delay|"             # Минимальная задержка
    "stimeout;5000000|"            # Таймаут сокета 5 секунд
    "max_delay;500000|"            # Максимальная задержка 0.5 сек
    "reorder_queue_size;0"         # Отключить очередь переупорядочивания
)
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8" # Отключаем логи ошибок декодирования в консоли

# --- КЛАСС ДЛЯ ЧТЕНИЯ ПОТОКА С АВТО-ПЕРЕПОДКЛЮЧЕНИЕМ ---
class VideoStream:
    def __init__(self, src):
        self.src = src
        self.capture = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.latest_frame = None
        self.last_frame_time = time.time()
        self.ret = False
        self.running = True
        self.lock = threading.Lock() # Блокировка для безопасного обмена кадрами между потоками
        
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.capture.read()
            
            if ret and frame is not None:
                # Если кадр получен успешно, обновляем его
                with self.lock:
                    self.latest_frame = frame
                    self.last_frame_time = time.time()
                    self.ret = True
            else:
                # Если произошла ошибка (bytestream -7, обрыв Wi-Fi)
                print("⚠️ Потеря связи или битый кадр. Переподключение...")
                self.capture.release()
                time.sleep(2) # Ждем 2 секунды перед переподключением
                
                if not self.running:
                    break
                    
                self.capture = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        with self.lock:
            return self.ret, self.latest_frame

    def release(self):
        self.running = False
        self.thread.join()
        self.capture.release()

# --- ОСНОВНАЯ ПРОГРАММА ---
if __name__ == "__main__":
    print(f"Загрузка модели {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print("Модель загружена. Подключение к камере...")

    vs = VideoStream(RTSP_URL)
    time.sleep(3.0) # Даем время на инициализацию камеры

    if not vs.ret:
        print("Ошибка: Не удалось получить первый кадр. Проверьте URL.")
        vs.release()
        exit()

    print("Поток запущен. Нажмите 'q' для выхода.")

    fps_start_time = time.time()
    fps_counter = 0
    fps_display = 0

    cv2.namedWindow("YOLO RTSP Detection", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = vs.read()
        
        # --- ЗАЩИТА ОТ БИТЫХ КАДРОВ ---
        # Если кадр не получен или он пустой, просто пропускаем итерацию
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        # Копируем кадр, чтобы YOLO не изменил его в фоновом потоке (защита от tearing)
        frame_to_process = frame.copy()

        # --- ОБРАБОТКА YOLO ---
        try:
            results = model(frame_to_process, conf=CONFIDENCE_THRESHOLD, imgsz=IMG_SIZE, verbose=False)
            
            # Рисуем рамки на кадре
            annotated_frame = results[0].plot()
        except Exception as e:
            print(f"Ошибка при обработке YOLO: {e}")
            annotated_frame = frame_to_process # Если YOLO упал, показываем оригинал

        # --- СЧИТАЕМ FPS ---
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_start_time = time.time()

        # Выводим FPS на экран
        cv2.putText(annotated_frame, f"FPS: {fps_display}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Показываем результат
        cv2.imshow("YOLO RTSP Detection", annotated_frame)

        # --- ВЫВОД В КОНСОЛЬ ---
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf > 0.60: 
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                print(f"[{time.strftime('%H:%M:%S')}] Найден: {class_name} (Conf: {conf:.2f})")

        # Выход по кнопке 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Очистка ресурсов
    vs.release()
    cv2.destroyAllWindows()