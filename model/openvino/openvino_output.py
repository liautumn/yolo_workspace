from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO(r"../pt/yolo11n.pt")  # load an official model

    # Export the model
    # OpenVINO ===> imgsz, half, dynamic, int8, nms, batch, data
    model.export(
        format='openvino',
        imgsz=(1024, 1024),
        half=True,
        dynamic=False,
        int8=False,
        nms=False,
        batch=1
    )
