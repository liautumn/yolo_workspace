from ultralytics import YOLO

if __name__ == '__main__':
    # Initialize the YOLO model
    model = YOLO(r"")

    # Define search space
    search_space = {
        "lr0": (1e-5, 1e-1)
    }

    # Tune hyperparameters on COCO8 for 30 epochs
    model.tune(
        data="",
        epochs=5,
        iterations=300,
        optimizer="AdamW",
        space=search_space,
        plots=False,
        save=False,
        val=False,
        project="",
        name="tune_01"
    )
