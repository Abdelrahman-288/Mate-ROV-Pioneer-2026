import os
import random
import shutil

BASE = r"C:\Users\abdel\OneDrive\Desktop\PIONEER CV Tasks\Task_2_1_Invasive_Species_Detection\dataset\yolo"

IMG_DIR = os.path.join(BASE, "images/train")
LBL_DIR = os.path.join(BASE, "labels/train")

imgs = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]
random.shuffle(imgs)

train = imgs[:int(0.7 * len(imgs))]
val = imgs[int(0.7 * len(imgs)):int(0.9 * len(imgs))]
test = imgs[int(0.9 * len(imgs)):]

# ✅ create folders safely
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(BASE, "images", split), exist_ok=True)
    os.makedirs(os.path.join(BASE, "labels", split), exist_ok=True)

def move(files, split):
    for f in files:
        src_img = os.path.join(IMG_DIR, f)
        dst_img = os.path.join(BASE, "images", split, f)

        lbl = f.replace(".jpg", ".txt")
        src_lbl = os.path.join(LBL_DIR, lbl)
        dst_lbl = os.path.join(BASE, "labels", split, lbl)

        # move image
        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)

        # move label if exists
        if os.path.exists(src_lbl):
            shutil.move(src_lbl, dst_lbl)
        else:
            print(f"[WARNING] Missing label: {lbl}")

# ❗ do NOT move train again
move(val, "val")
move(test, "test")

print("✅ Dataset split completed!")