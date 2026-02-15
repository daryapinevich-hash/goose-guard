# src/stereo.py
"""
Модуль стереокамеры с полной отказоустойчивостью.
Калибровка, depth map, fallback режимы.
"""

import cv2
import numpy as np
from pathlib import Path
import time
from src.constants import (
    LEFT_STREAM,
    RIGHT_STREAM,
    FALLBACK_VIDEO,
    BASELINE_CM,
    FOCAL_LENGTH,
    THREAT_DISTANCE,
)


class StereoCamera:
    def __init__(self, left_url=LEFT_STREAM, right_url=RIGHT_STREAM):
        self.left_cap = cv2.VideoCapture(left_url)
        self.right_cap = cv2.VideoCapture(right_url)
        self.last_mode = None
        self.calibrated = False

        self.fallback_cap = None
        self.fallback_pos = 0

        # Калибровочные матрицы (загружаем или None)
        self.load_calibration()

        # Стерео matcher
        self.stereo = cv2.StereoBM_create(numDisparities=16 * 8, blockSize=15)
        self.stereo.setMinDisparity(0)
        self.stereo.setSpeckleRange(15)
        self.stereo.setSpeckleWindowSize(100)

    def load_calibration(self):
        """Загрузка calibration.npz"""
        calib_path = Path("data/calibration.npz")
        if calib_path.exists():
            data = np.load(calib_path)
            self.mtxL, self.distL = data["mtxL"], data["distL"]
            self.mtxR, self.distR = data["mtxR"], data["distR"]
            self.R, self.T = data["R"], data["T"]
            self.calibrated = True
            print("✅ Calibration loaded from data/calibration.npz")
        else:
            print("⚠️  No calibration found. Run calibrate() first!")

    def save_calibration(self):
        """Сохранение калибровки"""
        np.savez(
            "data/calibration.npz",
            mtxL=self.mtxL,
            distL=self.distL,
            mtxR=self.mtxR,
            distR=self.distR,
            R=self.R,
            T=self.T,
        )
        self.calibrated = True
        print("💾 Calibration saved!")

    def _detect_mode(self):
        """Определение режима работы камер"""
        left_ok = self.left_cap.isOpened()
        right_ok = self.right_cap.isOpened()

        if left_ok and right_ok:
            return "both_alive"
        elif left_ok:
            return "mono_left"
        elif right_ok:
            return "mono_right"
        else:
            return "both_down"

    def _print_status(self, mode):
        if mode != self.last_mode:
            self.last_mode = mode
            status = {
                "both_alive": "✅ STEREO mode (оба живы)",
                "mono_left": "⚠️  Mono LEFT (правая отвалилась)",
                "mono_right": "⚠️  Mono RIGHT (левая отвалилась)",
                "both_down": "🚨 ALL CAMERAS DOWN - VIDEO FALLBACK",
            }
            print(status[mode])

    def get_frames(self):
        """Возвращает кадры + режим"""
        mode = self._detect_mode()
        self._print_status(mode)

        if mode == "both_alive":
            retL, frameL = self.left_cap.read()
            retR, frameR = self.right_cap.read()
            return frameL, frameR, mode if retL and retR else None

        elif mode == "mono_left":
            ret, frame = self.left_cap.read()
            return frame, None, mode if ret else None

        elif mode == "mono_right":
            ret, frame = self.right_cap.read()
            return frame, None, mode if ret else None

        else:  # both_down
            frame = None  # ← ДОБАВЬ ЭТО!

            if self.fallback_cap is None:
                print("📹 Initializing fallback VIDEO...")
                self.fallback_cap = cv2.VideoCapture(str(FALLBACK_VIDEO))

            if self.fallback_cap.isOpened():
                self.fallback_cap.set(cv2.CAP_PROP_POS_FRAMES, self.fallback_pos % 1000)
                ret, frame = self.fallback_cap.read()
                self.fallback_pos += 1

                if not ret or frame is None:
                    # Конец видео — сбрасываем
                    self.fallback_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.fallback_pos = 0
                    ret, frame = self.fallback_cap.read()
                    self.fallback_pos = 1

            if frame is None:  # Если видео не открылось
                print("❌ Fallback failed → черный экран")
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

            return frame, None, mode

    def get_depth(self, frameL, frameR=None):
        """Depth map по режиму"""
        if self.calibrated and frameR is not None:
            return self._stereo_depth(frameL, frameR)
        elif frameL is not None:
            return self._mono_depth(frameL)
        return None

    def _stereo_depth(self, frameL, frameR):
        """Стерео depth map"""
        # Предобработка
        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        # Rectify если калибровано
        if self.calibrated:
            grayL = cv2.undistort(grayL, self.mtxL, self.distL)
            grayR = cv2.undistort(grayR, self.mtxR, self.distR)

        # Disparity
        disparity = self.stereo.compute(grayL, grayR)
        depth = FOCAL_LENGTH * BASELINE_CM / 100 / (disparity + 1e-6)
        depth = np.clip(depth, 0, THREAT_DISTANCE * 2)
        return depth

    def _mono_depth(self, frame):
        """Запасная моно глубина по размеру bbox"""
        h, w = frame.shape[:2]
        depth_map = np.full((h, w), THREAT_DISTANCE * 1.5, dtype=np.float32)
        return depth_map

    def calibrate(self, left_video_path, right_video_path):
        """Полная калибровка шахматкой"""
        print("🎯 Starting calibration...")

        # Шахматка 9x6 (внутренние углы)
        CHECKERBOARD = (9, 6)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(
            -1, 2
        )

        objpoints = []
        imgpointsL, imgpointsR = [], []

        # Захват кадров из видео
        capL = cv2.VideoCapture(str(left_video_path))
        capR = cv2.VideoCapture(str(right_video_path))

        frame_count = 0
        while frame_count < 30:  # 30 кадров достаточно
            retL, frameL = capL.read()
            retR, frameR = capR.read()

            if not (retL and retR):
                break

            grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

            retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
            retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)

            if retL and retR:
                objpoints.append(objp)
                cornersL = cv2.cornerSubPix(
                    grayL, cornersL, (11, 11), (-1, -1), criteria
                )
                cornersR = cv2.cornerSubPix(
                    grayR, cornersR, (11, 11), (-1, -1), criteria
                )
                imgpointsL.append(cornersL)
                imgpointsR.append(cornersR)
                frame_count += 1

        capL.release()
        capR.release()

        if len(objpoints) < 10:
            print("❌ Need more chessboard images!")
            return False

        # Калибровка
        h, w = grayL.shape
        retL, self.mtxL, self.distL, rvecsL, tvecsL = cv2.calibrateCamera(
            objpoints, imgpointsL, (w, h), None, flags=cv2.CALIB_RATIONAL_MODEL
        )
        retR, self.mtxR, self.distR, rvecsR, tvecsR = cv2.calibrateCamera(
            objpoints, imgpointsR, (w, h), None, flags=cv2.CALIB_RATIONAL_MODEL
        )

        # Stereo калибровка
        flags = cv2.CALIB_FIX_INTRINSIC
        ret, self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            objpoints,
            imgpointsL,
            imgpointsR,
            self.mtxL,
            self.distL,
            self.mtxR,
            self.distR,
            (w, h),
            flags=flags,
            criteria=criteria,
        )

        self.save_calibration()
        return True


if __name__ == "__main__":
    """Тестовый запуск"""
    print("🎥 Testing StereoCamera...")
    stereo = StereoCamera()

    fps_time = time.time()
    frame_count = 0

    while True:
        frameL, frameR, mode = stereo.get_frames()

        if frameL is not None:
            frame_count += 1
            # Масштабируем для показа
            displayL = cv2.resize(frameL, (640, 480))

            # Depth если стерео
            if frameR is not None and stereo.calibrated:
                depth = stereo.get_depth(frameL, frameR)
                depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
                depth_vis = cv2.resize(depth_vis, (640, 480))
                displayL = np.hstack([displayL, depth_vis])

            cv2.imshow("Гусиный Страж", displayL)

            # FPS
            if time.time() - fps_time > 2.0:
                print(f"📹 FPS: {frame_count/2:.1f} | Mode: {mode}")
                fps_time = time.time()
                frame_count = 0

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    print("👋 Test finished")


def __del__(self):
    """Освобождение камер"""
    self.left_cap.release()
    self.right_cap.release()
