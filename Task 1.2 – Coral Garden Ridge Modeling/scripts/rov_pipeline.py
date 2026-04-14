import cv2
import os
import time
import numpy as np

# =========================
# CONFIG
# =========================
STREAM_SOURCE = 0  # replace with RTSP later
FRAME_DIR = "../frames"

os.makedirs(FRAME_DIR, exist_ok=True)

cap = cv2.VideoCapture(STREAM_SOURCE)

last_saved = time.time()
save_interval = 1.0  # seconds

prev_frame = None
saved_count = 0

# =========================
# BLUR DETECTION FUNCTION
# =========================
def is_blurry(frame, threshold=100):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap < threshold

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()

    if not ret:
        print("No stream yet — waiting...")
        time.sleep(1)
        continue

    cv2.imshow("ROV Live", frame)

    now = time.time()

    # SAVE ONLY GOOD FRAMES
    if now - last_saved > save_interval:

        if not is_blurry(frame):

            # avoid duplicates using simple difference
            if prev_frame is None or np.mean(cv2.absdiff(frame, prev_frame)) > 5:

                filename = f"{FRAME_DIR}/frame_{saved_count:05d}.jpg"
                cv2.imwrite(filename, frame)

                print("Saved:", filename)
                saved_count += 1
                prev_frame = frame.copy()
                last_saved = now

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()