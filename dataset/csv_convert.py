import json
import os
import pandas as pd

# 读取CSV文件
csv_file = 'D:/dataset/Intestinal Organoid Dataset/test_labels.csv'  # 替换为你的CSV文件路径
csv_data = pd.read_csv(csv_file)

# 创建COCO格式的基础结构
coco_format = {
    "images": [],
    "annotations": [],
    "categories": [{
        "id": 1,
        "name": "organoid",
        "supercategory": "cell"
    }]
}

annotation_id = 1  # 初始注释ID
image_id_mapping = {}  # 用于映射图像路径到COCO中的image_id
image_id = 1  # 初始图像ID

# 遍历CSV中的每一行并转换为COCO格式
for index, row in csv_data.iterrows():
    image_path = row[0]
    xmin, ymin, xmax, ymax, category = row[1:6]

    # 处理图像部分
    image_filename = os.path.basename(image_path)
    if image_path not in image_id_mapping:
        # 如果该图像尚未添加，则添加到COCO格式的images部分
        width = int(image_filename[9:12])
        height = int(image_filename[13:16])
        image_info = {
            "id": image_id,
            "file_name": image_filename,
            "width": width,  # 假设固定宽度300（根据图像分辨率设置）
            "height": height  # 假设固定高度300
        }
        coco_format["images"].append(image_info)
        image_id_mapping[image_path] = image_id
        image_id += 1

    # 处理注释部分（边界框）
    bbox = [xmin, ymin, xmax - xmin, ymax - ymin]  # COCO格式的边界框 (x, y, width, height)

    annotation = {
        "id": annotation_id,
        "image_id": image_id_mapping[image_path],
        "category_id": 1,  # 'organoid' 的类别ID为1
        "bbox": bbox,
        "area": (xmax - xmin) * (ymax - ymin),
        "segmentation": [],  # 如果没有提供多边形分割，这部分可以留空
        "iscrowd": 0  # 不是crowd对象
    }
    coco_format["annotations"].append(annotation)
    annotation_id += 1

# 保存为COCO格式的JSON文件
output_json_file = 'D:/dataset/Intestinal Organoid Dataset/annotations_coco_format.json'  # 替换为输出路径
with open(output_json_file, 'w') as json_file:
    json.dump(coco_format, json_file, indent=4)

print(f'COCO annotations saved to {output_json_file}')
