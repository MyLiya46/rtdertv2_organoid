import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# === 路径配置 ===
#annFile = r"D:\Workspace\Organoid_Tracking\tracking_labeled\organoid_labeled\train\annotations\instances_default.json"
#img_dir = r"D:\Workspace\Organoid_Tracking\tracking_labeled\organoid_labeled\train"

annFile = r"/home/ubuntu/emma_myers/tracking_labeled//organoid_labeled/train/annotations/instances_default.json"
img_dir = r"/home/ubuntu/emma_myers/tracking_labeled/organoid_labeled/train"

num_images_to_show = 5  # 仅可视化前 N 张图像

# === 读取 COCO JSON ===
with open(annFile, 'r', encoding='utf-8') as f:
    coco = json.load(f)

images = coco['images']
annotations = coco['annotations']
categories = {cat['id']: cat['name'] for cat in coco['categories']}

# === image_id -> annotation 列表映射 ===
imageid_to_anns = defaultdict(list)
for ann in annotations:
    imageid_to_anns[ann['image_id']].append(ann)

# === 可视化标注 ===
shown = 0
for img_info in images:
    file_name = img_info['file_name']
    image_id = img_info['id']
    img_path = os.path.join(img_dir, file_name)

    if not os.path.exists(img_path):
        print(f"❌ 图像 {img_path} 不存在，跳过")
        continue

    anns = imageid_to_anns[image_id]
    print(f"✅ 图像 {file_name} 共 {len(anns)} 个标注")

    # 加载图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ 无法读取图像 {img_path}，跳过")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 绘制标注
    for ann in anns:
        x, y, w, h = ann['bbox']
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)

        cat_name = categories.get(ann['category_id'], 'unknown')
        cv2.putText(img, cat_name, (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        segs = ann.get('segmentation', [])
        for seg in segs:
            pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # 显示图像
    plt.figure(figsize=(10, 10))
    plt.title(f"{file_name} - {len(anns)} objects")
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    shown += 1
    if shown >= num_images_to_show:
        break
