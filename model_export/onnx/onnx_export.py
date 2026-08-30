from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(r"yolo26l.pt")  # load an official model

    # Export the model
    # ONNX ===> imgsz(h,w), half, dynamic, simplify, opset, batch
    model.export(
        format='onnx',
        imgsz=(1280, 1280),
        half=True,
        dynamic=True,
        simplify=True,
        # batch=1,
        # opset=19,
    )
