import os
import shutil
from tqdm import tqdm  # 进度条库


def classify_yolo_data(root_dir, output_dir, class_names):
    """
    将YOLO格式的数据按照类别分类。

    :param root_dir: 数据根目录，包含images和labels文件夹
    :param output_dir: 输出目录，用于存放分类后的数据
    :param class_names: 类别名称列表，按顺序对应YOLO标签中的类别ID
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 创建每个类别的文件夹
    for class_name in class_names:
        class_image_dir = os.path.join(output_dir, class_name, 'images')
        class_label_dir = os.path.join(output_dir, class_name, 'labels')
        os.makedirs(class_image_dir, exist_ok=True)
        os.makedirs(class_label_dir, exist_ok=True)

    # 遍历images文件夹
    images_dir = os.path.join(root_dir, 'images')
    labels_dir = os.path.join(root_dir, 'labels')

    # 获取所有图片文件
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    # 记录已处理的文件，避免重复操作
    processed_files = set()

    # 遍历图片文件并显示进度条
    for image_name in tqdm(image_files, desc="Processing images", unit="image"):
        image_path = os.path.join(images_dir, image_name)
        label_name = image_name.replace('.jpg', '.txt')  # 假设图片是.jpg格式
        label_path = os.path.join(labels_dir, label_name)

        if not os.path.exists(label_path):
            continue  # 如果没有对应的标签文件，跳过

        # 检查是否已经处理过该文件
        if image_name in processed_files:
            continue

        # 读取标签文件
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # 提取图片中的所有类别
        classes_in_image = set()
        for line in lines:
            class_id = int(line.split()[0])  # YOLO格式的第一个值是类别ID
            if class_id < len(class_names):  # 确保类别ID在范围内
                classes_in_image.add(class_names[class_id])

        # 将图片和标签复制到每个类别的文件夹中
        for class_name in classes_in_image:
            class_image_dir = os.path.join(output_dir, class_name, 'images')
            class_label_dir = os.path.join(output_dir, class_name, 'labels')

            # 复制图片
            shutil.copy(image_path, os.path.join(class_image_dir, image_name))
            # 复制标签
            shutil.copy(label_path, os.path.join(class_label_dir, label_name))

        # 标记文件为已处理
        processed_files.add(image_name)

    print(f"数据分类完成！结果已保存到 {output_dir} 目录中。")


# 示例调用
if __name__ == '__main__':
    # 根目录路径（包含images和labels文件夹）
    root_dir = '/media/autumn/新加卷/zhenzhi/val'
    # 输出目录路径（用于存放分类后的数据）
    output_dir = '/media/autumn/新加卷/针织数据'
    # 类别名称列表（按顺序对应YOLO标签中的类别ID）
    class_names = [
        'BB', 'ZH', 'ZK', 'JK', 'ZZ', 'GS', 'ZW', 'DJ', 'PD', 'CS',
        'DW', 'HN', 'YW', 'FH', 'LZ', 'SYQ', 'BQ', 'DPD', 'MD', 'CH',
        'SD', 'SZ', 'ZS', 'FS', 'HT'
    ]

    # 调用函数进行分类
    classify_yolo_data(root_dir, output_dir, class_names)
