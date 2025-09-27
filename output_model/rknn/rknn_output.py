from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(r"D:\autumn\Documents\GitHub\ultralytics\ultralytics\yolo_workspace\鸟\训练\train2\weights\best.pt")  # load an official model

    # Export the model 导出到 RKNN 时，请确保使用 x86 Linux 机器
    # RKNN ===> imgsz(h,w), batch, name, device
    model.export(
        format='rknn',
        imgsz=(640, 640),
        name="rk3588",
        device=0,
        batch=1
    )
