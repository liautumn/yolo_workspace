from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    # model = YOLO("/home/autumn/Documents/ultralytics/my_workspace/针织/训练/train5_yolo11l/weights/last.pt")
    # model = YOLO("yolo11s.yaml").load("yolo11s.pt")
    model = YOLO("yolo11n.pt")
    # model = YOLO("yolo11s.yaml")

    # Train the model
    model.train(
        data="/media/autumn/新加卷/针织数据/data.yaml",
        epochs=20,
        batch=64,
        imgsz=640,
        workers=8,
        resume=False,
        project="针织/训练",
        # name="train"
    )
