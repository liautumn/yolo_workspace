from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO(r"D:\autumn\Documents\GitHub\ultralytics\ultralytics\yolo_workspace\鸟\训练\train2\weights\last.pt")
    # model = YOLO("yolo11s.yaml").load("yolo11s.pt")
    # model = YOLO("yolo11n.pt")
    # model = YOLO("yolo11s.yaml")

    # Train the model
    model.train(
        data=r"D:\autumn\Downloads\bird.v6i.yolov11\data.yaml",
        epochs=200,
        batch=48,
        imgsz=640,
        workers=8,
        resume=True, #从上次保存的检查点恢复训练
        save_period=10,
        project="鸟/训练",
        # name="train"
    )
