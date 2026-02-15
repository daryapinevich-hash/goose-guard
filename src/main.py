# src/main.py

from src.stereo import StereoCamera

stereo = StereoCamera()

# Калибровка (один раз)
# stereo.calibrate("chess_left.mp4", "chess_right.mp4")

while True:
    frameL, frameR, mode = stereo.get_frames()
    if frameL is not None:
        depth = stereo.get_depth(frameL, frameR)
        cv2.imshow("Depth", depth / 10)  # Визуализация
