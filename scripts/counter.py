from ultralytics import YOLO
import cv2
import os
import random

# =========================
# 1. Load model
# =========================
model = YOLO(r"C:\Users\abdel\OneDrive\Desktop\PIONEER CV Tasks\Task_2_1_Invasive_Species_Detection\runs\train\weights\best.pt")

# =========================
# 2. Load dataset folder
# =========================
folder = r"C:\Users\abdel\OneDrive\Desktop\PIONEER CV Tasks\Task_2_1_Invasive_Species_Detection\dataset\yolo\images\val"

# Get all images
files = [f for f in os.listdir(folder) if f.endswith(".jpg")]

# Pick 5 random images
selected_images = random.sample(files, min(5, len(files)))

# =========================
# 3. Process images
# =========================
for image_name in selected_images:

    image_path = os.path.join(folder, image_name)
    print(f"📷 Processing: {image_name}")

    img = cv2.imread(image_path)

    if img is None:
        print("❌ Failed to load image")
        continue

    # Run inference
    results = model(img, conf=0.5)

    count = 0

    # Draw detections
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            label = f"{model.names[cls]} {conf:.2f}"

            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

            count += 1

    # =========================
    # 4. Draw crab counter (TOP LEFT)
    # =========================
    counter_text = f"Crabs: {count}"

    cv2.putText(img, counter_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 3)

    # =========================
    # 5. Show image
    # =========================
    cv2.imshow("Detection", img)

    key = cv2.waitKey(0)  # wait for key

    # Press 'q' to exit early
    if key == ord('q'):
        break

cv2.destroyAllWindows()