import os
import json
import cv2
import matplotlib.pyplot as plt

# 配置路径
data_dir = r"D:\Workspace\Organoid_Tracking\tracking_labeled\organoid_labeled\train"

# 找出前 N 个 JSON 文件
json_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])[:5]

for json_file in json_files:
    json_path = os.path.join(data_dir, json_file)
    img_name = os.path.splitext(json_file)[0] + ".jpg"
    img_path = os.path.join(data_dir, img_name)

    # 读取图像
    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ 无法读取：{img_path}")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 读取 JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 支持两种格式：LabelMe(shapes) 或自定义(objects)
    objects = data.get('objects') or data.get('shapes', [])
    for obj in objects:
        if 'segmentation' in obj:
            seg = obj['segmentation']
        else:
            seg = obj.get('points', [])
        # 如果为嵌套列表，取第一个
        if seg and isinstance(seg[0][0], (list, tuple)):
            seg = seg[0]
        pts = [(int(x), int(y)) for x,y in seg]
        # 绘制多边形
        for i in range(len(pts)):
            pt1, pt2 = pts[i], pts[(i+1)%len(pts)]
            cv2.line(image, pt1, pt2, (0,255,0), 2)
    
    # 显示
    plt.figure(figsize=(8,6))
    plt.title(json_file)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
