from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO(r"D:\autumn\Documents\JetBrainsProjects\PycharmProjects\Yolo11\runs\detect\train\weights\best.pt")  # load a custom model

    # Validate the model
    validation_results = model.val(
        data=r"D:\autumn\Documents\JetBrainsProjects\PycharmProjects\Yolo11\训练数据\FaceDetection\data.yaml",  # 指定数据集配置文件的路径（如 coco8.yaml）。该文件包括指向训练和验证数据的路径、类名和类数。
        imgsz=640,  # 定义输入图像的尺寸。所有图像在处理前都会调整到这一尺寸。
        batch=64,  # 设置每批图像的数量。使用 -1 的自动批处理功能，它会根据 GPU 内存可用性自动调整。
        save_json=True,  # 如果为 True，将结果保存到 JSON 文件中，以便进一步分析或与其他工具集成。
        save_hybrid=True,  # 如果为 True，保存混合版本的标签，将原始注释与额外的模型预测相结合。
        conf=0.405,  # 设置检测的最小置信度阈值。置信度低于此阈值的检测将被丢弃。
        iou=0.5,  # 设置非最大抑制 (NMS) 的交叉重叠 (IoU) 阈值。有助于减少重复检测。
        max_det=300,  # 限制每幅图像的最大检测次数。在密度较高的场景中非常有用，可以防止检测次数过多。
        half=True,  # 是否进行半精度（FP16）计算，减少内存使用量，在提高速度的同时，将对精度的影响降至最低。
        device=[0],  # 指定验证设备（如 cpu, cuda:0 等）。可灵活利用 CPU 或 GPU 资源。
        dnn=False,  # 如果为 True，使用 OpenCV 的 DNN 模块进行 ONNX 模型推理，而不是 PyTorch 推理方法。
        plots=True,  # 当设置为 True 时，生成并保存预测结果与地面实况的对比图，以便对模型的性能进行可视化评估。
        rect=True,  # 如果为 True，使用矩形推理进行批处理，减少了填充，可能会提高速度和效率。
        split="val",  # 确定用于验证的数据集分割（val, test 或 train）。可灵活选择数据段进行性能评估。
        project=None,  # 保存验证输出的项目目录名称。
        name=None,  # 验证运行的名称。用于在项目文件夹内创建一个子目录，用于存储验证日志和输出结果。
    )

    print(validation_results.box.map)  # mAP50-95
    print(validation_results.box.map50)  # mAP50
    print(validation_results.box.map75)  # mAP75
    print(validation_results.box.maps)  # list of mAP50-95 for each category
