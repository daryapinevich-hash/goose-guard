# src/detector.py

from ultralytics import YOLO
import cv2
import numpy as np
from constants import MIN_CONFIDENCE, DETECT_SIZE


class GooseDetector:
    def __init__(self, model_path="yolov8n.pt", conf=MIN_CONFIDENCE):
        """
        YOLOv8 DETECTOR + TRACKER для гусей.
        Возвращает: [{'id':15, 'x1,y1,x2,y2', 'conf':0.85, 'pred_pos':(x,y)}, ...]
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.bird_class = 15  # COCO: bird
        self.track_history = {}  # {track_id: [(x,y,t), ...]} для предсказания

    def detect(self, frame):
        """TRACK MODE — ID + предсказания для ВСЕХ гусей"""
        results = self.model.track(
            frame,
            imgsz=DETECT_SIZE,
            conf=self.conf,
            persist=True,  # Держит track_id между кадрами
            tracker="bytetrack.yaml",  # Быстрый трекер
            verbose=False,
        )[0]

        tracks = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = (
                results.boxes.id.cpu().numpy().astype(int)
                if results.boxes.id is not None
                else []
            )
            confs = results.boxes.conf.cpu().numpy()

            for i, (box, track_id, conf) in enumerate(zip(boxes, track_ids, confs)):
                x1, y1, x2, y2 = map(int, box)
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                # ПРЕДСКАЗАНИЕ позиции через 0.5с (линейная экстраполация)
                pred_pos = self._predict_position(track_id, center_x, center_y)

                tracks.append(
                    {
                        "id": int(track_id),
                        "bbox": [x1, y1, x2, y2],
                        "center": (center_x, center_y),
                        "pred_pos": pred_pos,  # КУДА ЛЕТИТ через 0.5с
                        "conf": float(conf),
                        "size": (x2 - x1) * (y2 - y1),  # Площадь для threat_score
                        "velocity": self._get_velocity(track_id),  # px/s
                    }
                )

        return tracks

    def _predict_position(self, track_id, curr_x, curr_y):
        """Простое предсказание: скорость × 0.5с"""
        if track_id in self.track_history:
            history = self.track_history[track_id]
            if len(history) > 2:
                # Линейная регрессия по последним 3 точкам
                times = np.array([t for _, _, t in history[-3:]])
                xs = np.array([x for x, _, _ in history[-3:]])
                ys = np.array([y for _, y, _ in history[-3:]])

                # Скорость px/s
                vx = (xs[-1] - xs[0]) / (times[-1] - times[0])
                vy = (ys[-1] - ys[0]) / (times[-1] - times[0])

                pred_x = curr_x + vx * 0.5  # 0.5 сек вперед
                pred_y = curr_y + vy * 0.5
                return int(pred_x), int(pred_y)

        # Нет истории — текущая позиция
        self.track_history[track_id] = [(curr_x, curr_y, cv2.getTickCount())]
        return curr_x, curr_y

    def _get_velocity(self, track_id):
        """Скорость px/s"""
        if track_id in self.track_history and len(self.track_history[track_id]) > 1:
            history = self.track_history[track_id]
            dx = history[-1][0] - history[-2][0]
            dy = history[-1][1] - history[-2][1]
            dt = (history[-1][2] - history[-2][2]) / cv2.getTickFrequency()
            return (dx**2 + dy**2) ** 0.5 / dt if dt > 0 else 0
        return 0

    def update_history(self, tracks):
        """Обновляет историю для следующего кадра"""
        current_time = cv2.getTickCount()
        for track in tracks:
            track_id = track["id"]
            x, y = track["center"]
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append((x, y, current_time))

            # Храним только последние 30 кадров
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)

    def draw_birds(self, frame, tracks, color=(0, 255, 0), thickness=2):
        """Рисует треки + предсказания"""
        frame_out = frame.copy()
        for track in tracks:
            x1, y1, x2, y2 = map(int, track["bbox"])
            pred_x, pred_y = track["pred_pos"]
            track_id = track["id"]

            # 🟢 Bbox гуся
            cv2.rectangle(frame_out, (x1, y1), (x2, y2), color, thickness)

            # 🟡 Предсказанная точка (0.5с вперед)
            cv2.circle(frame_out, (pred_x, pred_y), 8, (0, 255, 255), -1)

            # 🔴 Линия до предсказания
            cv2.line(
                frame_out,
                (track["center"][0], track["center"][1]),
                (pred_x, pred_y),
                (0, 255, 255),
                2,
            )

            # ID + threat info
            label = f"ID:{track_id} c:{track['conf']:.1f}"
            cv2.putText(
                frame_out, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

        return frame_out
