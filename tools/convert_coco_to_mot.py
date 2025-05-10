"""
将COCO格式标注转换为MOT多目标跟踪格式(.txt)
"""

import json

def convert_json_to_mot(json_file_path, mot_file_path, id_range=None):
    # 读取 JSON 文件
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # 初始化 MOT 标签文件
    mot_labels = []

    # 遍历所有标注
    for annotation in data['annotations']:
        # 提取所需信息
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        bbox = annotation['bbox']
        iscrowd = annotation['iscrowd']
        ignore = annotation['ignore']
        track_id = str(int(annotation['track_id']) + 1)

        # 如果指定了 id_range，检查是否在范围内
        if id_range is not None and image_id not in id_range:
            continue

        # MOT 标签格式：frame, id, x, y, w, h, ignore, occluded, generated
        mot_label = [
            image_id,  # frame
            track_id,  # id
            bbox[0],  # x
            bbox[1],  # y
            bbox[2],  # w
            bbox[3],  # h
            ignore,  # ignore
            0,  # occluded
            0  # generated
        ]

        # 添加到 MOT 标签列表
        mot_labels.append(mot_label)

    # 将 MOT 标签写入文件
    with open(mot_file_path, 'w') as f:
        for label in mot_labels:
            f.write(','.join(map(str, label)) + '\n')

# 调用函数
json_file_path = 'D:/dataset/organoids_tracking/pancreatic_cancer/Y202312006_P8/annotations/instances_default.json'
mot_file_path = 'D:/dataset/organoids_tracking/pancreatic_cancer/Y202312006_P8/img_visu/label/gt.txt'
id_range = list(range(1, 43))  # 转换前100张图像的标签
convert_json_to_mot(json_file_path, mot_file_path, id_range)