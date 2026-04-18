from ultralytics import YOLO

def main():
    model = YOLO(
        r"C:\Users\abdel\OneDrive\Desktop\PIONEER CV Tasks\Task_2_1_Invasive_Species_Detection\runs\train\weights\best.pt"
    )

    metrics = model.val(
        data=r"C:\Users\abdel\OneDrive\Desktop\PIONEER CV Tasks\Task_2_1_Invasive_Species_Detection\data.yaml",
        imgsz=640,
        batch=8,
        conf=0.25,     # evaluation threshold (keep default-ish)
        iou=0.6,       # NMS IoU
        device="cuda",
        workers=0,     # keep 0 on Windows to avoid freeze
        plots=True     # saves confusion matrix, PR curves, etc.
    )

    print("\n📊 Evaluation Results:")
    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")

if __name__ == "__main__":
    main()