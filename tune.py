from ultralytics import YOLO

if __name__ == '__main__':
    # Initialize the YOLO model
    model = YOLO("/home/autumn/桌面/ultralytics/my_workspace/针织/训练/train3/weights/best.pt")

    # Define search space
    search_space = {
        "lr0": (1e-5, 1e-1)
    }

    # Tune hyperparameters on COCO8 for 30 epochs
    model.tune(
        data="/media/autumn/新加卷/zhenzhi/data.yaml",
        epochs=5,
        iterations=300,
        optimizer="AdamW",
        space=search_space,
        plots=False,
        save=False,
        val=False,
        project="针织/调参",
        name="tune3"
    )