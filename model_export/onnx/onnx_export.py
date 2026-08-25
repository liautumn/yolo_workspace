from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(r"yolo26n")  # load an official model

    # Export the model
    # ONNX ===> imgsz(h,w), half, dynamic, simplify, opset, batch
    model.export(
        format='onnx',
        imgsz=(640, 640),
        half=True,
        dynamic=True,
        simplify=True,
        # batch=1,
        # opset=19,
    )
