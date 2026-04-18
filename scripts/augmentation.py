import os
import cv2
import albumentations as A

IMG_DIR = r"C:\Users\abdel\OneDrive\Desktop\PIONEER CV Tasks\Task_2_1_Invasive_Species_Detection\dataset\all_data/images"
LBL_DIR = r"C:\Users\abdel\OneDrive\Desktop\PIONEER CV Tasks\Task_2_1_Invasive_Species_Detection\dataset\all_data/labels"
OUT_IMG = r"C:\Users\abdel\OneDrive\Desktop\PIONEER CV Tasks\Task_2_1_Invasive_Species_Detection\dataset\yolo/images/train"
OUT_LBL = r"C:\Users\abdel\OneDrive\Desktop\PIONEER CV Tasks\Task_2_1_Invasive_Species_Detection\dataset\yolo/labels/train"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.Blur(p=0.3),
    A.CoarseDropout(p=0.3)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

for img_name in os.listdir(IMG_DIR):
    if not img_name.endswith(".jpg"):
        continue

    img_path = os.path.join(IMG_DIR, img_name)
    lbl_path = os.path.join(LBL_DIR, img_name.replace(".jpg", ".txt"))

    image = cv2.imread(img_path)

    bboxes = []
    class_labels = []

    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                cls, x, y, w, h = map(float, line.strip().split())
                bboxes.append([x, y, w, h])
                class_labels.append(int(cls))

    for i in range(3):
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)

        new_img_name = f"{img_name.split('.')[0]}_aug_{i}.jpg"
        cv2.imwrite(os.path.join(OUT_IMG, new_img_name), augmented["image"])

        with open(os.path.join(OUT_LBL, new_img_name.replace(".jpg", ".txt")), "w") as f:
            for bbox, cls in zip(augmented["bboxes"], augmented["class_labels"]):
                f.write(f"{cls} {' '.join(map(str, bbox))}\n")