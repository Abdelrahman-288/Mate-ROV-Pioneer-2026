from ultralytics import YOLO
import cv2
import os
import random
import numpy as np

model = YOLO(r"C:\Users\abdel\OneDrive\Desktop\PIONEER CV Tasks\Task_2_1_Invasive_Species_Detection\runs\train\weights\best.pt")

folder = r"C:\Users\abdel\OneDrive\Desktop\PIONEER CV Tasks\Task_2_1_Invasive_Species_Detection\dataset\yolo\images\val"

files = [f for f in os.listdir(folder) if f.endswith(".jpg")]

selected = random.sample(files, min(5, len(files)))

for image_name in selected:
    path = os.path.join(folder, image_name)
    img = cv2.imread(path)

    if img is None:
        continue

    h, w = img.shape[:2]

    results = model(img, conf=0.5, iou=0.6)

    count = 0

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            bw = x2 - x1
            bh = y2 - y1
            area = bw * bh

            # 🔥 filters
            if area > 0.25 * w * h:
                continue

            roi = img[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            texture = np.var(gray)

            if texture < 300:
                continue

            cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
            count += 1

    cv2.putText(img, f"Crabs: {count}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    cv2.imshow("Result", img)

    if cv2.waitKey(0) == ord('q'):
        break

cv2.destroyAllWindows()