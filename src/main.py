# main.py - Гусиный Страж v1
import cv2
import numpy as np
import time
import threading
from pathlib import Path
from constants import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    MIN_CONFIDENCE,
    THREAT_DISTANCE,
    ALARM_SOUND_PATH,
)
from stereo import StereoCamera
from detector import GooseDetector
import pyaudio
import wave

detector = GooseDetector(conf=MIN_CONFIDENCE)


def play_alarm():
    """Тревожный звук"""
    try:
        wf = wave.open(str(ALARM_SOUND_PATH), "rb")
        p = pyaudio.PyAudio()
        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
        )
        data = wf.readframes(1024)
        while data:
            stream.write(data)
            data = wf.readframes(1024)
        stream.stop_stream()
        stream.close()
        p.terminate()
    except Exception as e:
        print(f"⚠️ Alarm failed: {e}")


def draw_crosshair(frame, cx, cy, size=30, thickness=2, color=(0, 255, 255)):
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, thickness)
    cv2.circle(frame, (cx, cy), size // 2, color, thickness)


def draw_radius_zone(frame, cx, cy, target_distance=None):
    pixel_radius = min(SCREEN_WIDTH // 4, SCREEN_HEIGHT // 4)
    cv2.circle(frame, (cx, cy), pixel_radius, (0, 0, 255), 3)
    if target_distance is not None:
        dist_text = f"Target: {target_distance:.0f}m"
        color = (0, 255, 0) if target_distance < 50 else (0, 0, 255)
    else:
        dist_text = "No target"
        color = (128, 128, 128)
    cv2.putText(
        frame, dist_text, (cx - 70, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
    )


class StableTargetTracker:
    def __init__(self):
        self.target_pos = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        self.last_seen = time.time()
        self.max_lost_time = 2.0

    def update(self, tracks, frame_shape):
        if tracks:
            best_track = max(tracks, key=lambda t: t.get("size", 0) * t.get("conf", 0))
            pred_x, pred_y = best_track.get(
                "pred_pos", best_track.get("center", (0, 0))
            )
            h, w = frame_shape[:2]
            pred_x = max(50, min(w - 50, pred_x))
            pred_y = max(50, min(h - 50, pred_y))
            self.target_pos = (int(pred_x), int(pred_y))
            self.last_seen = time.time()
            return True
        return False

    def get_target(self, frame_shape):
        if time.time() - self.last_seen > self.max_lost_time:
            h, w = frame_shape[:2]
            self.target_pos = (w // 2, h // 2)
        return self.target_pos


if __name__ == "__main__":
    print("🎯 Goose Guard v2.1 - FULL SYSTEM!")

    # ИНИЦИАЛИЗАЦИЯ
    stereo = StereoCamera()
    stable_tracker = StableTargetTracker()
    alarm_active = False
    last_alarm_time = 0

    fps_time = time.time()
    frame_count = 0

    while True:
        frameL, frameR, mode = stereo.get_frames()

        if frameL is not None:
            frame_count += 1
            orig_shape = frameL.shape

            # DETECTION
            tracks = detector.detect(frameL)
            detector.update_history(tracks)

            # TRACKING
            has_target = stable_tracker.update(tracks, orig_shape)
            target_pos = stable_tracker.get_target(orig_shape)

            # DRAWING
            displayL = detector.draw_birds(frameL, tracks, color=(0, 255, 0))
            cx, cy = target_pos
            draw_crosshair(displayL, cx, cy)

            # DISTANCE
            target_distance = None
            if (
                frameR is not None
                and hasattr(stereo, "calibrated")
                and stereo.calibrated
            ):
                try:
                    depth_map = stereo.get_depth(frameL, frameR)
                    depth_h, depth_w = depth_map.shape
                    depth_cx = int(cx * depth_w / orig_shape[1])
                    depth_cy = int(cy * depth_h / orig_shape[0])
                    depth_cx = max(0, min(depth_w - 1, depth_cx))
                    depth_cy = max(0, min(depth_h - 1, depth_cy))
                    target_distance = float(depth_map[depth_cy, depth_cx])
                except:
                    pass

            # FALLBACK DISTANCE BY SIZE
            if target_distance is None and tracks:
                best_track = max(tracks, key=lambda t: t.get("size", 0))
                size = best_track.get("size", 1)
                if size > 5000:
                    target_distance = 15
                elif size > 1000:
                    target_distance = 30
                elif size > 200:
                    target_distance = 50
                else:
                    target_distance = 80

            draw_radius_zone(displayL, cx, cy, target_distance)

            # ТРЕВОГА <30м
            threat_detected = (
                bool(tracks) and target_distance is not None and target_distance < 30
            )

            if (
                threat_detected
                and not alarm_active
                and (time.time() - last_alarm_time > 5)
            ):
                print("🚨 THREAT <30m! ALARM ON!")
                threading.Thread(target=play_alarm, daemon=True).start()
                alarm_active = True
                last_alarm_time = time.time()
            elif len(tracks) == 0 or target_distance >= 30:
                alarm_active = False

            # STATUS
            if tracks:
                best_track = max(
                    tracks, key=lambda t: t.get("size", 0) * t.get("conf", 0)
                )
                status = f"ID:{best_track.get('id', 'N/A')} C:{best_track.get('conf', 0):.1f}"
                cv2.putText(
                    displayL,
                    status,
                    (15, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 255),
                    2,
                )
                alarm_status = "ALARM!" if threat_detected else "OK"
                cv2.putText(
                    displayL,
                    f"Tracks: {len(tracks)} {alarm_status}",
                    (15, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255) if threat_detected else (255, 255, 255),
                    2,
                )
            else:
                cv2.putText(
                    displayL,
                    "SEARCHING...",
                    (15, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    2,
                )

            # RESIZE + DEPTH
            displayL = cv2.resize(displayL, (SCREEN_WIDTH, SCREEN_HEIGHT))
            if frameR is not None and stereo.calibrated:
                try:
                    depth = stereo.get_depth(frameL, frameR)
                    depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
                    depth_vis = cv2.resize(depth_vis, (SCREEN_WIDTH, SCREEN_HEIGHT))
                    displayL = np.hstack([displayL, depth_vis])
                except:
                    pass

            cv2.imshow("Goose Guard v2.1", displayL)

            # FPS
            if time.time() - fps_time > 5.0:
                real_fps = frame_count / (time.time() - fps_time)
                print(
                    f"FPS: {real_fps:.1f} | Tracks: {len(tracks)} | Dist: {target_distance or 0:.1f}m | Alarm: {'ON' if threat_detected else 'OFF'}"
                )
                fps_time = time.time()
                frame_count = 0

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            stable_tracker.last_seen = 0
            print("Tracker reset!")

    cv2.destroyAllWindows()
    print("Goose Guard stopped")
