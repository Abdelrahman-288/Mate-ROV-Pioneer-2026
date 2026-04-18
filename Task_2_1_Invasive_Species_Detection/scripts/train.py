from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")  # 🔥 better model

    model.train(
        data=r"C:\Users\abdel\OneDrive\Desktop\PIONEER CV Tasks\Task_2_1_Invasive_Species_Detection\data.yaml",
        epochs=100,             # 🔥 more training
        imgsz=640,
        batch=8,
        device="cuda",
        workers=0,
        project=r"C:\Users\abdel\OneDrive\Desktop\PIONEER CV Tasks\Task_2_1_Invasive_Species_Detection\runs",
        name="train"
    )

if __name__ == "__main__":
    main()