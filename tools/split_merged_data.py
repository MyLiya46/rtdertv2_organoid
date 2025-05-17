import os
import json
import shutil
import random
from pathlib import Path

# 配置
# json_path = Path(r"D:\Workspace\Organoid_Tracking\tracking_labeled\organoid_labeled\annotations\instances_default.json")
# save_dir = Path(r"D:\Workspace\Organoid_Tracking\tracking_labeled\organoid_labeled")

json_path = Path(r"/home/ubuntu/emma_myers/tracking_labeled/organoid_labeled/annotations/instances_default.json")
save_dir = Path(r"/home/ubuntu/emma_myers/tracking_labeled/organoid_labeled")


train_dir = save_dir / "train"
val_dir = save_dir / "val"
val_ratio = 0.2  # 验证集比例

# 创建目录
for d in [train_dir, val_dir]:
    (d / "annotations").mkdir(parents=True, exist_ok=True)

# 加载 COCO JSON
with open(json_path, 'r', encoding='utf-8') as f:
    coco = json.load(f)
images = coco['images']
annotations = coco['annotations']
categories = coco['categories']
licenses = coco.get('licenses', [])
info = coco.get('info', {})

# 打乱并划分
random.seed(42)
random.shuffle(images)
split = int(len(images) * (1 - val_ratio))
train_images = images[:split]
val_images = images[split:]

def save_split(subset_images, subset_name):
    subset_ids = {img['id'] for img in subset_images}
    subset_annotations = [ann for ann in annotations if ann['image_id'] in subset_ids]
    # 构建子 COCO
    coco_subset = {
        "info": info,
        "licenses": licenses,
        "categories": categories,
        "images": subset_images,
        "annotations": subset_annotations
    }
    # 保存 JSON
    out_json = save_dir / subset_name / "annotations" / "instances_default.json"
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(coco_subset, f, indent=4, ensure_ascii=False)
    # 复制文件
    for img in subset_images:
        fname = img['file_name']
        src_img = save_dir / fname
        src_json = save_dir / (Path(fname).stem + ".json")
        dst_img = save_dir / subset_name / fname
        dst_json = save_dir / subset_name / (Path(fname).stem + ".json")
        shutil.copy(src_img, dst_img)
        if src_json.exists():
            shutil.copy(src_json, dst_json)

# 保存 train 和 val
save_split(train_images, "train")
save_split(val_images, "val")

print(f"划分完成：训练集 {len(train_images)} 张，验证集 {len(val_images)} 张")
