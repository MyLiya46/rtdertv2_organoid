"""
整理分散的医学图像数据集到统一目录。
"""

import os
import shutil
import json

def extract_files(source_root, target_folder):
    # 确保目标文件夹存在
    os.makedirs(target_folder, exist_ok=True)

    # 遍历主文件夹中的所有文件夹
    for case in os.listdir(source_root):
        case_path = os.path.join(source_root, case)

        for folder in os.listdir(case_path):
            folder_path = os.path.join(case_path, folder)

            # 检查是否为文件夹
            if os.path.isdir(folder_path):
                # 遍历第一级文件夹
                for subfolder in os.listdir(folder_path):
                    subfolder_path = os.path.join(folder_path, subfolder)

                    # 检查是否为0文件夹
                    if os.path.isdir(subfolder_path) and subfolder == '0':
                        # 遍历第二级文件夹中的crop文件夹
                        crop_folder_path = os.path.join(subfolder_path, 'crop')
                        if os.path.isdir(crop_folder_path):
                            for file_name in os.listdir(crop_folder_path):
                                file_path = os.path.join(crop_folder_path, file_name)

                                # 只处理图像和JSON文件
                                if file_name.endswith(('.jpg', '.jpeg', '.png', '.json')):
                                    # 创建新文件名，添加原始路径作为前缀
                                    new_file_name = f"{case}-{file_name}"
                                    target_file_path = os.path.join(target_folder, new_file_name)

                                    # 复制文件到目标文件夹
                                    shutil.copy(file_path, target_file_path)
                                    print(f"Copied: {file_path} to {target_file_path}")

                                    # 如果是JSON文件，修改其内容
                                    if file_name.endswith('.json'):
                                        with open(target_file_path, 'r', encoding='utf-8') as json_file:
                                            data = json.load(json_file)

                                        # 修改name字段
                                        if 'info' in data and 'name' in data['info']:
                                            original_name = data['info']['name']
                                            # 这里可以根据需要修改name字段，例如添加前缀
                                            data['info']['name'] = f"{case}-{original_name}"

                                        # 将修改后的数据写回文件
                                        with open(target_file_path, 'w', encoding='utf-8') as json_file:
                                            json.dump(data, json_file, ensure_ascii=False, indent=4)
                                            print(f"Updated JSON name field: {target_file_path}")

# 设置源文件夹和目标文件夹
source_root = 'D:/dataset/organoids_det_raw/breast_cancer/'  # 替换为你的主文件夹路径
target_folder = 'D:/dataset/organoids_det_raw/breast/all'  # 替换为你想要保存文件的目标文件夹路径

# 执行提取
extract_files(source_root, target_folder)