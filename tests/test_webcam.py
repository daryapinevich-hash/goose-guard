# tests\test_webcam.py
import cv2

for i in range(5):  # Пробуем индексы 0-4
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Камера #{i} работает!")
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Cam {i}", frame)
            cv2.waitKey(1000)  # 1 секунда
        cap.release()
    else:
        print(f"❌ Камера #{i} не найдена")
cv2.destroyAllWindows()
