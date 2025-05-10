import os
import platform

# 设置统一的数据路径
if platform.system() == 'Windows':
    os.environ['DATA_PATH'] = r"D:\Workspace\Organoid_Tracking\tracking_labeled\all"
else:
    os.environ['DATA_PATH'] = "/home/ubuntu/emma_myers/organoid_tracking/rtdertv2_organoid/data/all"
