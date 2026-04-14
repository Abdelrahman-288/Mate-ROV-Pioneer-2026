import cv2
import os

INPUT = "../frames"
OUTPUT = "../outputs/targets"

os.makedirs(OUTPUT, exist_ok=True)

def is_square(contour):
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    return len(approx) == 4

targets = []

for f in os.listdir(INPUT):
    img = cv2.imread(os.path.join(INPUT, f))
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if is_square(c):
            x, y, w, h = cv2.boundingRect(c)
            targets.append((f, x, y, w, h))

print("Detected targets:", len(targets))