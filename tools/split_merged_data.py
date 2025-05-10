"""
带标注的数据集划分
保持图像与JSON标注的对应关系
按比例随机分割训练/测试集（默认8:2）
支持任意嵌套目录结构处理
"""
import os
import shutil
import random

def split_data(source_folder, train_folder, test_folder, train_ratio=0.8):
    # 确保训练和测试文件夹存在
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # 存储图像及其对应的JSON文件
    img_json_pairs = []

    # 遍历源文件夹以收集图像和JSON文件对
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # 检查文件是否为图像或JSON，并确保每个图像都有对应的JSON文件
            if file.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                json_file = file.replace(os.path.splitext(file)[1], '.json')
                json_path = os.path.join(root, json_file)
                if os.path.exists(json_path):
                    img_json_pairs.append((img_path, json_path))
            elif file.endswith('.json'):
                img_file = file.replace('.json', '.jpg')
                img_path = os.path.join(root, img_file)
                if os.path.exists(img_path):
                    img_json_pairs.append((img_path, os.path.join(root, file)))

    # 随机打乱文件对列表
    random.shuffle(img_json_pairs)

    # 计算训练集大小
    train_size = int(len(img_json_pairs) * train_ratio)

    # 分割训练集和测试集文件对
    train_files = img_json_pairs[:train_size]
    test_files = img_json_pairs[train_size:]

    # 复制文件对到训练和测试文件夹
    for img_path, json_path in train_files:
        train_img_name = os.path.basename(img_path)
        train_json_name = os.path.basename(json_path)
        shutil.copy(img_path, os.path.join(train_folder, train_img_name))
        shutil.copy(json_path, os.path.join(train_folder, train_json_name))

    for img_path, json_path in test_files:
        test_img_name = os.path.basename(img_path)
        test_json_name = os.path.basename(json_path)
        shutil.copy(img_path, os.path.join(test_folder, test_img_name))
        shutil.copy(json_path, os.path.join(test_folder, test_json_name))

    print(f"Copied {len(train_files)} file pairs to {train_folder}")
    print(f"Copied {len(test_files)} file pairs to {test_folder}")

# 设置源文件夹、训练文件夹和测试文件夹的路径
source_folder = 'D:/dataset/organoids_det_raw/pancreatic/all/'  # 替换为你的源文件夹路径
train_folder = 'D:/dataset/organoids_det_raw/pancreatic/train'  # 替换为你的训练文件夹路径
test_folder = 'D:/dataset/organoids_det_raw/pancreatic/test'  # 替换为你的测试文件夹路径

# 执行数据分割
split_data(source_folder, train_folder, test_folder)

