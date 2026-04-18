import cv2
import os

video_path = "../input_video/rov.mp4"
output_folder = "../frames"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_id = 0
saved_id = 0

fps_skip = 5  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % fps_skip == 0:
        filename = f"{output_folder}/frame_{saved_id:05d}.jpg"
        cv2.imwrite(filename, frame)
        saved_id += 1

    frame_id += 1

cap.release()

print(f"Saved {saved_id} frames")