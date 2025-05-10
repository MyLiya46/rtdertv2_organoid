import json
import os
import shutil
from pathlib import Path

# Windows 本地路径
datasets = [
    {
        "name": "pancreatic",
        "image_dir": r"D:\Workspace\Organoid_Tracking\tracking_labeled\pancreatic_cancer_labeled\img_1",
        "annotation": r"D:\Workspace\Organoid_Tracking\tracking_labeled\pancreatic_cancer_labeled\annotations\MOT\instances_default.json"
    },
    {
        "name": "stomach",
        "image_dir": r"D:\Workspace\Organoid_Tracking\tracking_labeled\stomach_cancer_labeled\img_1",
        "annotation": r"D:\Workspace\Organoid_Tracking\tracking_labeled\stomach_cancer_labeled\annotations\MOT\instances_default.json"
    }
]

# 合并输出路径（Windows 本地）
output_dir = Path(r"D:\Workspace\Organoid_Tracking\tracking_labeled\all")
merged_img_dir = output_dir / "img_1"
merged_ann_path = output_dir / "annotations\MOT\instances_default.json"
merged_img_dir.mkdir(parents=True, exist_ok=True)
merged_ann_path.parent.mkdir(parents=True, exist_ok=True)

# 初始化 COCO 格式结构
merged = {
    "images": [],
    "annotations": [],
    "categories": [],
}

image_id = 0
annotation_id = 0

for ds in datasets:
    with open(ds["annotation"], "r") as f:
        data = json.load(f)

    # 统一类别
    if not merged["categories"]:
        merged["categories"] = data["categories"]
    else:
        assert data["categories"] == merged["categories"], "❌ 类别不一致"

    old_to_new_image_id = {}
    for img in data["images"]:
        old_id = img["id"]
        new_filename = f"{ds['name']}_{img['file_name']}"
        src_path = os.path.join(ds["image_dir"], img["file_name"])
        dst_path = os.path.join(merged_img_dir, new_filename)
        shutil.copy(src_path, dst_path)

        new_img = img.copy()
        new_img["id"] = image_id
        new_img["file_name"] = new_filename
        merged["images"].append(new_img)

        old_to_new_image_id[old_id] = image_id
        image_id += 1

    for ann in data["annotations"]:
        new_ann = ann.copy()
        new_ann["id"] = annotation_id
        new_ann["image_id"] = old_to_new_image_id[ann["image_id"]]
        merged["annotations"].append(new_ann)
        annotation_id += 1

with open(merged_ann_path, "w") as f:
    json.dump(merged, f)