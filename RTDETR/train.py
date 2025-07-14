from ultralytics import RTDETR

if __name__ == '__main__':
    # Load a COCO-pretrained RT-DETR-l model
    model = RTDETR("../model/pt/rtdetr-l.pt")

    # Display model information (optional)
    model.info()

    # results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

    # Train the model
    model.train(
        data="coco8.yaml",
        epochs=20,
        batch=64,
        imgsz=640,
        workers=8,
        resume=False,
        # project="针织/训练",
        # name="train3"
    )