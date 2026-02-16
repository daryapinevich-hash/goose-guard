from constants import THREAT_DISTANCE, THREAT_HIGH, BASELINE_CM
from tracker import GooseTracker  # Multi-target


class ThreatManager:
    def __init__(self):
        self.trackers = {}  # {track_id: GooseTracker}
        self.threats = []  # Список угроз по приоритету

    def update(self, bboxes, depth_map, frame_shape):
        """Обновляет threat_list из сырых данных"""
        threats = []

        for bbox in bboxes:
            track_id = self._assign_tracker(bbox)
            if track_id:
                tracker = self.trackers[track_id]

                # 1. Предсказание позиции через 0.5с
                pred_x, pred_y = tracker.predict_future()

                # 2. Глубина в точке предсказания
                depth = self._get_depth_at(pred_x, pred_y, depth_map)

                # 3. Threat Score = f(size, speed, distance, direction)
                score = self._calculate_threat(bbox, depth, tracker.velocity)

                threats.append(
                    {
                        "id": track_id,
                        "bbox": bbox,
                        "pred_pos": (pred_x, pred_y),
                        "depth_m": depth,
                        "speed_px_s": tracker.velocity,
                        "threat_score": score,
                        "priority": "HIGH" if score > THREAT_HIGH else "MEDIUM",
                    }
                )

        # Сортировка по threat_score
        self.threats = sorted(threats, key=lambda t: t["threat_score"], reverse=True)
        return self.threats

    def get_target_for_turret(self):
        """ТОЛЬКО №1 угроза для турели"""
        return self.threats[0] if self.threats else None
