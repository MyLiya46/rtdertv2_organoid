"""
文件筛选与复制
按命名模式（L-0001.jpg）筛选JPG图片
跨目录复制符合条件的文件
用于数据集初步整
"""

import os
import shutil

# 定义源目录和目标目录
source_dir = "D:/dataset/organoids_tracking/stomach_cancer/20240530070952907/"
destination_dir = "D:/dataset/organoids_tracking/stomach_cancer/img/"

# 确保目标目录存在
os.makedirs(destination_dir, exist_ok=True)

# 遍历源目录中的文件
for filename in os.listdir(source_dir):
    # 检查文件是否为jpg格式，并且命名格式为L-0001
    if filename.endswith(".jpg") and filename.count('-') == 1 and filename.startswith("L-"):
        # 构建完整的文件路径
        source_file = os.path.join(source_dir, filename)
        destination_file = os.path.join(destination_dir, filename)
        # 复制文件到目标目录
        shutil.copy(source_file, destination_file)

print("筛选完成！")

