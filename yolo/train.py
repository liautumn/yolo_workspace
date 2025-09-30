from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    # model = YOLO(r"D:\autumn\Documents\GitHub\ultralytics\ultralytics\yolo_workspace\train_output\bird\train\train3\weights\last.pt")
    # model = YOLO("yolo11s.yaml").load("yolo11s.pt")
    model = YOLO("yolo11s.pt")
    # model = YOLO("yolo11s.yaml")

    # Train the model
    model.train(
        data=r"D:\autumn\Downloads\bird.v6i.yolov11\data.yaml",
        epochs=250,
        batch=32,
        imgsz=640,
        workers=8,
        resume=False, #从上次保存的检查点恢复训练
        save_period=10, #保存间隔
        project="../train_output/bird/train",
        name="train02"
    )
