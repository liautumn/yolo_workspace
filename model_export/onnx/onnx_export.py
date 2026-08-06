from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(r"/Users/autumn/IdeaProjects/ultralytics/ultralytics/yolo_workspace/train_output/rock-paper-scissors/train_01/weights/best.pt")  # load an official model

    # Export the model
    # ONNX ===> imgsz(h,w), half, dynamic, simplify, opset, batch
    model.export(
        format='onnx',
        imgsz=(640, 640),
        half=True,
        dynamic=False,
        simplify=True,
        batch=1,
        # opset=19,
    )
