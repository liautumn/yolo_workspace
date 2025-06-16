from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(r"../pt/yolo11s-seg.pt")  # load an official model

    # Export the model
    # ONNX ===> imgsz(h,w), half, dynamic, simplify, opset, batch
    model.export(
        format='onnx',
        imgsz=(1024, 1024),
        half=True,
        dynamic=False,
        simplify=True,
        batch=1
    )
