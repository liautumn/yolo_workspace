import os
from collections import defaultdict


def count_images_per_class(output_dir):
    """
    计算每个类别文件夹中的图片数量，并保存结果到文件。

    :param output_dir: 输出目录，包含按类别分类的文件夹
    """
    # 初始化类别计数器
    class_counts = defaultdict(int)

    # 遍历输出目录中的每个类别文件夹
    for class_name in os.listdir(output_dir):
        class_dir = os.path.join(output_dir, class_name)
        if not os.path.isdir(class_dir):
            continue  # 跳过非文件夹

        # 检查是否存在images文件夹
        images_dir = os.path.join(class_dir, 'images')
        if not os.path.exists(images_dir):
            continue  # 跳过没有images文件夹的类别

        # 统计images文件夹中的图片数量
        image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]  # 假设图片是.jpg格式
        class_counts[class_name] = len(image_files)

    # 保存类别数量到文件
    count_file_path = os.path.join(output_dir, 'class_counts.txt')
    with open(count_file_path, 'w') as f:
        for class_name, count in class_counts.items():
            f.write(f'{class_name}: {count}\n')

    print(f"类别数量统计完成！结果已保存到 {count_file_path}。")


# 示例调用
if __name__ == '__main__':
    # 输出目录路径（包含按类别分类的文件夹）
    output_dir = '/media/autumn/新加卷/针织数据/所有类别'

    # 调用函数统计类别数量
    count_images_per_class(output_dir)
