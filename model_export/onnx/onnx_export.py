from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(r"D:\autumn\Documents\GitHub\ultralytics\ultralytics\yolo_workspace\鸟\训练\train2\weights\best.pt")  # load an official model

    # Export the model
    # ONNX ===> imgsz(h,w), half, dynamic, simplify, opset, batch
    model.export(
        format='onnx',
        imgsz=(640, 640),
        half=True,
        dynamic=False,
        simplify=True,
        batch=1,
        opset=19,
    )
