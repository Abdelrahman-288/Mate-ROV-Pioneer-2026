import os
import shutil

BASE = r"C:\Users\abdel\OneDrive\Desktop\PIONEER CV Tasks\Task_2_1_Invasive_Species_Detection"

YOLO_IMG = os.path.join(BASE, "dataset", "yolo", "images", "train")
YOLO_LBL = os.path.join(BASE, "dataset", "yolo", "labels", "train")
RUNS_DIR = os.path.join(BASE, "runs")

print("🧹 Cleaning old synthetic data...")

# =========================
# 1. Delete synthetic images + labels
# =========================
deleted = 0

for file in os.listdir(YOLO_IMG):
    if file.startswith("synthetic_"):
        img_path = os.path.join(YOLO_IMG, file)
        lbl_path = os.path.join(YOLO_LBL, file.replace(".jpg", ".txt"))

        if os.path.exists(img_path):
            os.remove(img_path)

        if os.path.exists(lbl_path):
            os.remove(lbl_path)

        deleted += 1

print(f"✅ Deleted {deleted} synthetic samples")

# =========================
# 2. Delete runs folder
# =========================
if os.path.exists(RUNS_DIR):
    shutil.rmtree(RUNS_DIR)
    print("✅ Deleted runs folder")
else:
    print("ℹ️ No runs folder found")

print("🎯 Cleanup completed successfully!")