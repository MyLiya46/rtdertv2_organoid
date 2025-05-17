import os
import json
import shutil
from glob import glob
from PIL import Image

# 输入路径（格式1和格式2的根目录）
input_dirs = [
    r"D:\Workspace\Organoid_Tracking\tracking_labeled\HF202406001-P3",
    r"D:\Workspace\Organoid_Tracking\tracking_labeled\RX2023405002-P2",
    r"D:\Workspace\Organoid_Tracking\tracking_labeled\stomach_cancer_labeled\img_1",
    r"D:\Workspace\Organoid_Tracking\tracking_labeled\pancreatic_cancer_labeled\img_1"
]

# 输出路径
output_root = r"D:\Workspace\Organoid_Tracking\tracking_labeled\organoid_labeled"
os.makedirs(output_root, exist_ok=True)
os.makedirs(os.path.join(output_root, "annotations"), exist_ok=True)

# 初始化 COCO 格式字典
coco = {
    "info": {
        "year": 2023,
        "version": "2.5.3",
        "description": "COCO Label Conversion",
        "contributor": "CVHub",
        "url": "https://github.com/CVHub520/X-AnyLabeling",
        "date_created": "2025-02-08"
    },
    "licenses": [
        {
            "id": 1,
            "url": "https://www.gnu.org/licenses/gpl-3.0.html",
            "name": "GNU GENERAL PUBLIC LICENSE Version 3"
        }
    ],
    "categories": [
        {"id": 1, "name": "organoid", "supercategory": ""},
        {"id": 2, "name": "background", "supercategory": ""}
    ],
    "images": [],
    "annotations": []
}

image_id = 1
annotation_id = 1

for input_dir in input_dirs:
    # 遍历所有图片
    for img_path in sorted(glob(os.path.join(input_dir, '*.jpg'))):
        # 定义新的文件名
        filename = f"L-{image_id:04d}.jpg"
        dst_img = os.path.join(output_root, filename)
        shutil.copy(img_path, dst_img)

        # JSON 原文件
        json_path = os.path.splitext(img_path)[0] + '.json'
        new_json = os.path.join(output_root, f"L-{image_id:04d}.json")
        shapes = []
        width, height = None, None

        if os.path.exists(json_path):
            shutil.copy(json_path, new_json)
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 图像尺寸
            width = data.get('imageWidth') or data.get('width')
            height = data.get('imageHeight') or data.get('height')
            if width is None or height is None:
                img = Image.open(img_path)
                width, height = img.size

            # 两种格式：LabelMe(shapes) 或 自定义(objects)
            if 'shapes' in data:
                shapes = data['shapes']
            elif 'objects' in data:
                # 将 objects 转为统一格式
                shapes = []
                for obj in data['objects']:
                    shapes.append({
                        'label': obj.get('category', obj.get('label', 'organoid')),
                        'points': obj.get('segmentation', []),
                    })

        # images 列表
        coco['images'].append({
            'id': image_id,
            'file_name': filename,
            'width': width,
            'height': height,
            'license': 0,
            'flickr_url': '',
            'coco_url': '',
            'date_captured': ''
        })

        # annotations 列表
        for shape in shapes:
            pts = shape['points']
            if not pts or len(pts) < 3:
                continue
            # 若 points 为嵌套列表，展开
            if isinstance(pts[0][0], list) or isinstance(pts[0][0], tuple):
                pts = pts[0]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            min_x, min_y, max_x, max_y = min(xs), min(ys), max(xs), max(ys)
            bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
            area = bbox[2] * bbox[3]
            segmentation = [[coord for p in pts for coord in p]]

            coco['annotations'].append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': 1,
                'bbox': bbox,
                'area': area,
                'iscrowd': 0,
                'ignore': 0,
                'track_id': -1,
                'segmentation': segmentation
            })
            annotation_id += 1

        image_id += 1

# 保存 COCO JSON
out_path = os.path.join(output_root, 'annotations', 'instances_default.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(coco, f, indent=4, ensure_ascii=False)

print(f"完成：共 {image_id-1} 张图，{annotation_id-1} 个标注。 输出文件：{out_path}")
