from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("/home/autumn/IdeaProjects/ultralytics/ultralytics/yolo_workspace/train_output/bird/train/train_01/weights/last.pt")

    # Train the model
    model.train(
        data=r"/home/autumn/Downloads/bird.yolo/data.yaml",
        epochs=250,
        batch=32,
        imgsz=640,
        workers=8,
        resume=True, #从上次保存的检查点恢复训练
        save_period=20, #保存间隔
        project="/home/autumn/IdeaProjects/ultralytics/ultralytics/yolo_workspace/train_output/bird/train",
        name="train_01"
    )
