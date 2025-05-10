"""
从文件夹中每隔5张抽取一张图片（用于降采样）。
"""

import os
import shutil

# 设置源文件夹和目标文件夹
source_folder = 'D:/dataset/organoids_tracking/pancreatic_cancer/V-20240625052734644/raw'
destination_folder = 'D:/dataset/organoids_tracking/pancreatic_cancer/V-20240625052734644/img'

# 确保目标文件夹存在
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 获取源文件夹中所有文件的列表
files = os.listdir(source_folder)

# 过滤出图片文件，这里假设图片文件以.jpg结尾
images = [file for file in files if file.endswith('.jpg')]

# 每隔5张提取一张图片
for i in range(0, len(images), 5):
    # 构建完整的文件路径
    src_path = os.path.join(source_folder, images[i])
    dst_path = os.path.join(destination_folder, images[i])

    # 复制文件
    shutil.copy(src_path, dst_path)

print('每隔5张图片提取完成。')