# src/constants.py
"""
Центральное хранилище констант, путей и ресурсов.

Содержит:
- Параметры камер
- Параметры микрофонов
- Пути
- baseline, threat_dist
"""

from pathlib import Path

# Определяем корень проекта:
PROJECT_ROOT = Path(__file__).parent.parent
RESOURCES_PATH = PROJECT_ROOT / "resources"


# Пути к ресурсам
SOUNDS_PATH = RESOURCES_PATH / "sounds"
IMAGES_PATH = RESOURCES_PATH / "images"

# Пути к конкретным файлам
ALARM_SOUND_PATH = SOUNDS_PATH / "Sound_14909.mp3"

# Параметры телефонов (RTSP по Wi-Fi)
PHONE_LEFT_IP = "192.168.1.10"
PHONE_RIGHT_IP = "192.168.1.11"
RTSP_PORT = 8080


LEFT_STREAM = 0  # Веб-камера ноутбука
RIGHT_STREAM = None
# LEFT_STREAM = f"rtsp://{PHONE_LEFT_IP}:{RTSP_PORT}/video"
# RIGHT_STREAM = f"rtsp://{PHONE_RIGHT_IP}:{RTSP_PORT}/video"

# Логирование
LAST_CAMERA_MODE = None

# Fallback режимы
FALLBACK_MODES = {
    "both_alive": "stereo",  # Оба телефона живы
    "left_down": "mono_right",  # Левая сдохла
    "right_down": "mono_left",  # Правая сдохла
    "both_down": "video_fallback",  # Оба мертвы
}

# Fallback при отвале камеры
FALLBACK_VIDEO = RESOURCES_PATH / "images" / "test_geese.mp4"

# Аудио параметры
SAMPLE_RATE = 44100
GOOSE_FREQ_MIN = 1000  # Гц
GOOSE_FREQ_MAX = 4000

# Турель Arduino
ARDUINO_PORT = "COM3"  # Windows /dev/ttyUSB0 Linux
SERVO_PAN_PIN = 9
SERVO_TILT_PIN = 10
LASER_PIN = 11

# YOLO
MIN_CONFIDENCE = 0.6
GOOSE_CLASS_ID = 15  # "bird" в COCO (дообучить на гусях)

# Threat scoring
MAX_THREAT_SCORE = 1.0
THREAT_LOW = 0.3
THREAT_MEDIUM = 0.6
THREAT_HIGH = 0.9
THREAT_DISTANCE = 50

# STEREO_PARAMS
BASELINE_CM = 30.0
FOCAL_LENGTH = 800

# GUI
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768
