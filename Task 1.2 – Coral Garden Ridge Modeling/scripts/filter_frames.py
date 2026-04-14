import cv2
import os
import numpy as np

INPUT = "../frames"
OUTPUT = "../frames_filtered"

os.makedirs(OUTPUT, exist_ok=True)

def is_blurry(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < 120

prev = None
saved = 0

for f in sorted(os.listdir(INPUT)):
    path = os.path.join(INPUT, f)
    img = cv2.imread(path)

    if img is None:
        continue

    if is_blurry(img):
        continue

    if prev is not None:
        diff = np.mean(cv2.absdiff(img, prev))
        if diff < 5:
            continue

    cv2.imwrite(f"{OUTPUT}/frame_{saved:05d}.jpg", img)
    prev = img
    saved += 1

print("Filtered frames:", saved)