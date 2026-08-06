import os
import random
import shutil
from PIL import Image

# ========== 配置参数 ==========
source_dir = r"D:\autumn\Downloads\CCPD2020\ccpd_green\val"  # CCPD原始数据集路径
output_dir = "CCPD_YOLO"            # 输出YOLO格式数据集路径
split_ratios = {"train": 0.7, "val": 0.2, "test": 0.1}  # 数据集划分比例
class_name = "license_plate"       # 类别名称（YOLO格式需要对应类别ID）
# =============================

# 创建YOLO标准目录结构
os.makedirs(os.path.join(output_dir, "images/train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "images/val"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "images/test"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels/train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels/val"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels/test"), exist_ok=True)

# 获取所有图片文件并随机打乱
all_files = [f for f in os.listdir(source_dir) if f.endswith(".jpg")]
random.shuffle(all_files)

# 计算分割点
n_total = len(all_files)
n_train = int(n_total * split_ratios["train"])
n_val = int(n_total * split_ratios["val"])

# 文件分组
train_files = all_files[:n_train]
val_files = all_files[n_train:n_train+n_val]
test_files = all_files[n_train+n_val:]

def process_dataset(files, dataset_type):
    for file in files:
        # 解析边界框坐标（来自CCPD文件名格式）
        parts = file.split("-")
        bbox_str = parts[2].split("_")
        xmin, ymin = map(int, bbox_str[0].split("&"))
        xmax, ymax = map(int, bbox_str[1].split("&"))

        # 读取图片尺寸
        img_path = os.path.join(source_dir, file)
        with Image.open(img_path) as img:
            width, height = img.size

        # 转换为YOLO格式（归一化坐标）
        x_center = (xmin + xmax) / (2 * width)
        y_center = (ymin + ymax) / (2 * height)
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height

        # 写入标签文件
        label_file = file.replace(".jpg", ".txt")
        with open(os.path.join(output_dir, "labels", dataset_type, label_file), "w") as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        # 复制图片文件
        shutil.copy(img_path, os.path.join(output_dir, "images", dataset_type, file))

# 处理各数据集
process_dataset(train_files, "train")
process_dataset(val_files, "val")
process_dataset(test_files, "test")

print(f"数据集生成完成！\n总样本数：{n_total}\n"
      f"训练集：{len(train_files)} | 验证集：{len(val_files)} | 测试集：{len(test_files)}")
