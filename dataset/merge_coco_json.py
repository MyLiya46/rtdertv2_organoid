import json
import os
import numpy as np
from collections import defaultdict

# ISAT标注数据路径
ISAT_FOLDER = "D:/dataset/organoids_det/pancreatic_cancer/Y202312006/img/"
# 图像所在的路径
IMAGE_FOLDER = "D:/dataset/organoids_det/pancreatic_cancer/Y202312006/img"
# COCO格式的JSON文件存放路径
# 可以自己指定，这里是直接定义在图像文件夹下
COCO_PATH = os.path.join("D:/dataset/organoids_det/pancreatic_cancer/Y202312006/", "coco.json")

# 定义类别名称与ID号的映射
# 需要注意的是，不需要按照ISAT的classesition.txt里面的定义来
# 可以选择部分自己需要的类别， ID序号也可以重新填写(从0开始)
category_mapping = {"Cell": 1, "Organoid": 2, "DiedOrganoid": 3, "Bubble": 4, "Tissue": 5}
# 定义COCO格式的字典
# - "info"/"description" 里面写一下，你这个是什么的数据集
coco = {
    "info": {
        "description": "Color Block Segmentation",
        "version": "1.0",
        "year": 2023,
        "contributor": "",
        "date_created": ""
    },
    "images": [],
    "annotations": [],
    "categories": []
}
# 填写annotations栏目
for class_name, class_id in category_mapping.items():
    coco["categories"].append({"id": class_id, "name": class_name, "supercategory": ""})
# 图像序号
image_id = 1
# 标注序号
annotation_id = 1

# 获取所有JSON文件的名称
json_files = {file.split('.')[0] for file in os.listdir(IMAGE_FOLDER) if file.endswith('.json')}

# 获取所有图像文件的名称
image_files = {file.split('.')[0] for file in os.listdir(IMAGE_FOLDER) if file.endswith(('.jpg', '.png'))}

# 找出没有对应JSON文件的图像文件
orphan_images = image_files - json_files

# 删除没有对应JSON文件的图像
for image in orphan_images:
    image_path = os.path.join(IMAGE_FOLDER, image + '.png')  # 假设图像文件是png格式
    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"Deleted image without JSON: {image_path}")

# 遍历所有的ISAT文件夹
for filename in os.listdir(ISAT_FOLDER):
    # 判断是否为ISAT格式数据
    if not filename.endswith(".json"):
        continue
    # 载入ISAT数据
    with open(os.path.join(ISAT_FOLDER, filename), "r") as f:
        isat = json.load(f)
    # 获取图像名称
    image_filename = isat["info"]["name"]

    # 填写文件路径
    image_path = os.path.join(IMAGE_FOLDER, image_filename)
    image_info = {
        "id": image_id,
        "file_name": image_filename,
        "width": isat["info"]["width"],
        "height": isat["info"]["height"]
    }
    # 添加图像信息
    coco["images"].append(image_info)
    # 遍历标注信息
    for annotation in isat["objects"]:
        # 获取类别名称
        category_name = annotation["category"]
        # 位置类别名称(选择跳过)
        if category_name not in category_mapping:
            # print(f"未知类别名称: {category_name}")
            continue
        # 获取类别ID
        category_id = category_mapping[category_name]
        bbox = annotation["bbox"]
        bbox[2] = bbox[2]-bbox[0]
        bbox[3] = bbox[3] - bbox[1]
        # 提取分割信息
        segmentation = annotation["segmentation"]
        segmentation = np.uint32(segmentation)
        # 转换为一行的形式 [[x1, y1, w, h, ..., w, h]]
        segmentation = [(segmentation.reshape(-1)).tolist()]
        # 提取面积信息
        area = annotation["area"]
        # 定义标注信息
        annotation_info = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "segmentation": segmentation,
            "area": area,
            "iscrowd": 0
        }
        # 追加到annotations列表
        coco["annotations"].append(annotation_info)
        # 标注编号自增1
        annotation_id += 1
    image_id += 1
# 保存coco格式
with open(COCO_PATH, "w") as f:
    json.dump(coco, f, indent=4)
