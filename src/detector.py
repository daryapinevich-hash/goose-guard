# src/detector.py

from ultralytics import YOLO
import cv2
import numpy as np
from constants import MIN_CONFIDENCE, DETECT_SIZE


class GooseDetector:
    def __init__(self, model_path="yolov8n.pt", conf=MIN_CONFIDENCE):
        """
        Инициализация YOLOv8 детектора для птиц (класс 15 в COCO).
        model_path: yolov8n.pt (быстрый) или yolov8s.pt (точнее).
        conf: порог уверенности (0.5 = баланс скорость/точность).
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.bird_class = 15  # COCO: bird

    def detect(self, frame):
        results = self.model(
            frame, imgsz=DETECT_SIZE, conf=MIN_CONFIDENCE, verbose=False
        )[
            0
        ]  # conf=0.01! imgsz=320
        birds = []

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()

            # РИСУЕМ ВСЕ ОБЪЕКТЫ (не только птиц!)
            for i, (box, cls_id, conf) in enumerate(zip(boxes, classes, confs)):
                #    print(f"📦 #{i} класс={cls_id} conf={conf:.2f}")
                birds.append(box.tolist())  # ← ВСЕ в birds!

        # print(f"🔍 ИТОГО ОБЪЕКТОВ: {len(birds)}")
        return birds

    def draw_birds(self, frame, birds, color=(0, 255, 0), thickness=2):
        """
        Рисует зеленые рамки на птицах (для stereo.py).
        Возвращает: аннотированный кадр.
        """
        frame_out = frame.copy()
        for bbox in birds:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame_out, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                frame_out,
                "Bird",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                thickness,
            )
        return frame_out
