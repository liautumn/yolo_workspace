from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolo11s.pt")
    # model = YOLO("yolo11s.yaml")
    # model = YOLO("yolo11s.yaml").load("yolo11s.pt")

    # Train the model
    model.train(
        data=r"",
        epochs=250,
        batch=32,
        imgsz=640,
        workers=8,
        resume=False, #从上次保存的检查点恢复训练
        save_period=10, #保存间隔
        project="../train_output/bird/train",
        name="train_01"
    )
