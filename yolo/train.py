from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolo26s.pt")

    # Train the model
    model.train(
        data=r"data.yaml",
        epochs=250,
        batch=32,
        imgsz=640,
        workers=8,
        resume=True, #从上次保存的检查点恢复训练
        save_period=20, #保存间隔
        project="/Users/autumn/IdeaProjects/ultralytics/ultralytics/yolo_workspace/train_output/train-01",
        name="train_01"
    )
