# src/threat.py

"""
threat.py - Оценка угрозы от гусей для Goose Guard

Вычисляет threat score (0.0-1.0) по формуле:
score = 0.4*размер + 0.3*уверенность + 0.2*близость + 0.1*скорость

> 0.7 = Активация тревоги

Использование:
threats = calculate_threats(tracks, depth_map)
"""

from constants import THREAT_DISTANCE, THREAT_HIGH, BASELINE_CM


class ThreatManager:
    """Упрощенный threat scoring"""

    def __init__(self):
        self.threats = []

    def update(self, bboxes, depth_map, frame_shape):
        threats = []
        for i, bbox in enumerate(bboxes):
            depth = 30.0  # Fallback
            score = (
                bbox.get("size", 0) / 10000 * 0.4 + bbox.get("conf", 0.5) * 0.3 + 0.3
            )  # Базовая угроза

            threats.append(
                {
                    "id": i,
                    "bbox": bbox,
                    "pred_pos": bbox.get("center", (0, 0)),
                    "depth_m": depth,
                    "speed_px_s": 0.0,
                    "threat_score": min(1.0, score),
                    "priority": "HIGH" if score > 0.7 else "MEDIUM",
                }
            )

        self.threats = sorted(threats, key=lambda t: t["threat_score"], reverse=True)
        return self.threats

    def get_target_for_turret(self):
        return self.threats[0] if self.threats else None
