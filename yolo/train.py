from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolo26l.pt")

    # Train the model
    model.train(
        data=r"",
        device='cuda:0',
        epochs=200,
        batch=16,
        imgsz=1024,
        workers=8,
        resume=False, #从上次保存的检查点恢复训练
        save_period=10, #保存间隔
        project="/Users/autumn/IdeaProjects/ultralytics/ultralytics/yolo_workspace/train_output/xxx",
        name="train_01"
    )
