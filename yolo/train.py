from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolo26s.pt")

    # Train the model
    model.train(
        data=r"/Users/autumn/IdeaProjects/ultralytics/ultralytics/yolo_workspace/rock-paper-scissors.v14i.yolo26/data.yaml",
        device='mps',
        epochs=50,
        batch=10,
        imgsz=640,
        workers=8,
        resume=False, #从上次保存的检查点恢复训练
        save_period=10, #保存间隔
        project="/Users/autumn/IdeaProjects/ultralytics/ultralytics/yolo_workspace/train_output/rock-paper-scissors",
        name="train_01"
    )
