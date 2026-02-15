"""
Центральное хранилище констант, путей и ресурсов.

Содержит:
- Параметры камер
- Параметры микрофонов
- Пути
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


# Screen dimensions
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768
SCREEN_CENTER = (SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
# SCREEN_TITLE =
