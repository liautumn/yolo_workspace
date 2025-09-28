import os
import shutil
import random
from tqdm import tqdm  # 进度条库

def split_data(root_dir, output_dir, train_ratio=0.8):
    """
    将所有类别的数据按比例分配到 train 和 val 文件夹中。

    :param root_dir: 数据根目录，包含多个类别文件夹（如 BB、BQ 等），每个类别文件夹下包含 images 和 labels 文件夹
    :param output_dir: 输出目录，包含 train 和 val 文件夹
    :param train_ratio: 训练集比例，默认为 0.8
    """
    # 创建输出目录
    train_images_dir = os.path.join(output_dir, 'train', 'images')
    train_labels_dir = os.path.join(output_dir, 'train', 'labels')
    val_images_dir = os.path.join(output_dir, 'val', 'images')
    val_labels_dir = os.path.join(output_dir, 'val', 'labels')

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # 获取所有类别文件夹
    categories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    # 用于记录已复制的文件，避免重复
    copied_files = set()

    for category in categories:
        # 获取当前类别的 images 和 labels 文件夹路径
        images_dir = os.path.join(root_dir, category, 'images')
        labels_dir = os.path.join(root_dir, category, 'labels')

        # 获取当前类别的所有图片文件
        image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        total_files = len(image_files)

        # 随机打乱文件列表
        random.shuffle(image_files)

        # 计算训练集和验证集的分界点
        split_index = int(total_files * train_ratio)

        # 分配训练集
        for image_name in tqdm(image_files[:split_index], desc=f"Copying {category} train data", unit="file"):
            image_path = os.path.join(images_dir, image_name)
            label_name = image_name.replace('.jpg', '.txt')
            label_path = os.path.join(labels_dir, label_name)

            # 如果文件已经存在，则跳过
            if image_name in copied_files:
                continue

            # 复制图片和标签到训练集
            shutil.copy(image_path, os.path.join(train_images_dir, image_name))
            shutil.copy(label_path, os.path.join(train_labels_dir, label_name))

            # 记录已复制的文件
            copied_files.add(image_name)

        # 分配验证集
        for image_name in tqdm(image_files[split_index:], desc=f"Copying {category} val data", unit="file"):
            image_path = os.path.join(images_dir, image_name)
            label_name = image_name.replace('.jpg', '.txt')
            label_path = os.path.join(labels_dir, label_name)

            # 如果文件已经存在，则跳过
            if image_name in copied_files:
                continue

            # 复制图片和标签到验证集
            shutil.copy(image_path, os.path.join(val_images_dir, image_name))
            shutil.copy(label_path, os.path.join(val_labels_dir, label_name))

            # 记录已复制的文件
            copied_files.add(image_name)

    print(f"数据分配完成！训练集和验证集已保存到 {output_dir} 目录中。")

# 示例调用
if __name__ == '__main__':
    # 数据根目录（包含多个类别文件夹，如 BB、BQ 等）
    root_dir = '/media/autumn/新加卷/针织数据/所有类别'
    # 输出目录（包含 train 和 val 文件夹）
    output_dir = '/media/autumn/新加卷/针织数据/训练8.2'

    # 调用函数进行数据分配
    split_data(root_dir, output_dir, train_ratio=0.8)