import os
import json
import cv2

INPUT_DIR = r"C:\Users\abdel\OneDrive\Desktop\PIONEER CV Tasks\Task_2_1_Invasive_Species_Detection\dataset\all_data\images"
OUTPUT_DIR = r"C:\Users\abdel\OneDrive\Desktop\PIONEER CV Tasks\Task_2_1_Invasive_Species_Detection\dataset\all_data\labels"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in os.listdir(INPUT_DIR):
    if file.endswith(".json"):
        json_path = os.path.join(INPUT_DIR, file)

        with open(json_path) as f:
            data = json.load(f)

        # ✅ safer way to get image path
        img_name = os.path.basename(data["imagePath"])
        img_path = os.path.join(INPUT_DIR, img_name)

        img = cv2.imread(img_path)

        # ❌ FIX: handle missing images
        if img is None:
            print(f"[ERROR] Cannot read image: {img_path}")
            continue

        h, w = img.shape[:2]

        txt_path = os.path.join(OUTPUT_DIR, file.replace(".json", ".txt"))

        with open(txt_path, "w") as out:
            for shape in data["shapes"]:

                # only your class
                if shape["label"] != "green_crab":
                    continue

                points = shape["points"]

                # ❌ FIX: handle polygons (not just 2 points)
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]

                x1, x2 = min(x_coords), max(x_coords)
                y1, y2 = min(y_coords), max(y_coords)

                # YOLO format
                xc = ((x1 + x2) / 2) / w
                yc = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                out.write(f"0 {xc} {yc} {bw} {bh}\n")

print("✅ Conversion completed!")