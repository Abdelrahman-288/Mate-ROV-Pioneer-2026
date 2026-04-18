from ultralytics import YOLO
import cv2
import numpy as np

# =========================
# 1. Load trained model
# =========================
model = YOLO(r"C:\Users\abdel\OneDrive\Desktop\PIONEER CV Tasks\Task_2_1_Invasive_Species_Detection\runs\train\weights\best.pt")

# =========================
# 2. Start camera
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("🎥 Press 'q' to quit")

# =========================
# 🔥 Counter control
# =========================
display_count = 0
decay_timer = 0
DECAY_DELAY = 60  # ~2 seconds (assuming ~30 FPS)

# =========================
# 3. Real-time loop
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    h, w = frame.shape[:2]

    # =========================
    # 4. Run inference
    # =========================
    results = model(frame, conf=0.5, iou=0.6)

    count = 0

    # =========================
    # 5. Process detections
    # =========================
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            bw = x2 - x1
            bh = y2 - y1
            area = bw * bh

            # ❌ Filter large false detections
            if area > 0.25 * (w * h):
                continue

            # ❌ Texture filter (removes fake detections)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            texture = np.var(gray)

            if texture < 300:
                continue

            # Draw box
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

            count += 1

    # =========================
    # 🔥 Counter logic (ONLY CHANGE)
    # =========================
    if count > display_count:
        display_count = count   # 🚀 instant increase
        decay_timer = 0

    elif count < display_count:
        decay_timer += 1

        if decay_timer > DECAY_DELAY:
            display_count = count   # ⏳ delayed decrease
            decay_timer = 0
    else:
        decay_timer = 0

    # =========================
    # 6. Display counter
    # =========================
    cv2.putText(frame, f"Crabs: {display_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 3)

    # =========================
    # 7. Show frame
    # =========================
    cv2.imshow("🦀 Live Crab Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# 8. Cleanup
# =========================
cap.release()
cv2.destroyAllWindows()