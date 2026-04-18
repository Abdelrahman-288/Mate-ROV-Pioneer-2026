import cv2
import os
import random

SRC_IMG = r"C:\Users\abdel\OneDrive\Desktop\PIONEER CV Tasks\Task_2_1_Invasive_Species_Detection\dataset\yolo\images\train"
SRC_LBL = r"C:\Users\abdel\OneDrive\Desktop\PIONEER CV Tasks\Task_2_1_Invasive_Species_Detection\dataset\yolo\labels\train"

OUT_IMG = SRC_IMG
OUT_LBL = SRC_LBL

for i in range(300):
    base_name = random.choice(os.listdir(SRC_IMG))
    base_path = os.path.join(SRC_IMG, base_name)
    base = cv2.imread(base_path)

    if base is None:
        continue

    h, w = base.shape[:2]
    label_lines = []

    for _ in range(random.randint(1, 5)):
        crab_name = random.choice(os.listdir(SRC_IMG))
        crab_path = os.path.join(SRC_IMG, crab_name)
        crab_img = cv2.imread(crab_path)

        if crab_img is None:
            continue

        ch, cw = crab_img.shape[:2]

        # 🔥 resize crab (VERY IMPORTANT)
        scale = random.uniform(0.3, 0.6)
        new_w = int(cw * scale)
        new_h = int(ch * scale)

        crab_img = cv2.resize(crab_img, (new_w, new_h))

        if new_w >= w or new_h >= h:
            continue

        x = random.randint(0, w - new_w)
        y = random.randint(0, h - new_h)

        # 🔥 blend instead of hard paste
        roi = base[y:y+new_h, x:x+new_w]
        alpha = 0.7
        blended = cv2.addWeighted(roi, 1 - alpha, crab_img, alpha, 0)
        base[y:y+new_h, x:x+new_w] = blended

        # label
        xc = (x + new_w/2) / w
        yc = (y + new_h/2) / h
        bw = new_w / w
        bh = new_h / h

        label_lines.append(f"0 {xc} {yc} {bw} {bh}")

    name = f"synthetic_{i}.jpg"
    cv2.imwrite(os.path.join(OUT_IMG, name), base)

    with open(os.path.join(OUT_LBL, name.replace(".jpg", ".txt")), "w") as f:
        f.write("\n".join(label_lines))

print("✅ Synthetic dataset generated!")