import torch
from torch import nn
from torchvision.transforms import transforms
from PIL import Image, ImageDraw, ImageFont
import sys
import datetime
import shutil
from pathlib import Path
import numpy as np
import os
import argparse
from tqdm import tqdm
import colorsys
import csv

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig
from tracker.deep_eiou import Deep_EIoU

def setup_experiment(args):
    """创建实验目录结构并返回路径"""
    experiment_name = Path(args.output_dir).name
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    base_output_dir = Path("output/exp_track")
    output_dir = base_output_dir / f"{experiment_name}_{timestamp}"
    args.output_dir = str(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    # 子文件夹
    (output_dir / "det").mkdir(parents=True, exist_ok=True)
    (output_dir / "tra").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "configs").mkdir(parents=True, exist_ok=True)

    # 配置文件保存
    if args.config and Path(args.config).exists():
        shutil.copy(args.config, output_dir/"configs"/Path(args.config).name)
    
    return output_dir

def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},0,0,0\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(
                frame=frame_id,
                id=track_id + 1,
                x1=round(float(x1), 1),
                y1=round(float(y1), 1),
                w=round(float(w), 1),
                h=round(float(h), 1),
                s=round(float(score), 2)
            )
                f.write(line)

def generate_colors(color_id, num_colors=200):
    """
    根据给定的颜色ID生成RGB颜色元组。

    :param color_id: 颜色的唯一标识符（例如，bounding box的ID）。
    :param num_colors: 可用颜色的总数，用于确定颜色的多样性。
    :return: 一个RGB颜色元组 (int, int, int)。
    """
    # 将color_id映射到[0, num_colors-1]区间内
    hue = (color_id % num_colors) / num_colors
    saturation = 1.0  # 饱和度设置为1，表示最饱和的颜色
    value = 1.0  # 亮度设置为1，表示最亮的颜色
    # colorsys.hsv_to_rgb将HSV转换为RGB颜色
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    # 将RGB颜色从0-1范围转换为0-255范围
    return tuple(int(x * 255) for x in rgb)


def load_gt_for_frame(gt_file, orig_size, target_size):
    # scaled gt dict
    gt_dict = {}
    scale_x = target_size[0] / orig_size[0]
    scale_y = target_size[1] / orig_size[1]
    
    with open(gt_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            frame_id = int(row[0])
            x, y, w, h = map(float, row[2:6])
            if frame_id not in gt_dict:
                gt_dict[frame_id] = []
            gt_dict[frame_id].append((x * scale_x, y * scale_y, (x + w) * scale_x, (y + h) * scale_y))
    
    return gt_dict



class ImageReader:
    def __init__(self,orig_size=(3072, 2048), resize=(2400, 1600), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transform = transforms.Compose([
            # transforms.Resize((resize, resize)) if isinstance(resize, int) else transforms.Resize(
            #     (resize[0], resize[1])),
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean, std=std),
        ])
        self.orig_size = orig_size
        self.resize = resize
        self.pil_img = None  # 保存最近一次读取的图片的pil对象

    def __call__(self, image_path):
        """
        读取图片
        """
        self.pil_img = Image.open(image_path).convert('RGB').resize(self.resize)
        return self.transform(self.pil_img).unsqueeze(0)


class Model(nn.Module):
    def __init__(self, config=None, ckpt="") -> None:
        super().__init__()
        self.cfg = YAMLConfig(config, resume=ckpt)
        if ckpt:
            checkpoint = torch.load(ckpt, map_location='cpu', weights_only=True)
            if 'ema' in checkpoint:
                state = checkpoint['ema']['module']
            else:
                state = checkpoint['model']
        else:
            raise AttributeError('only support resume to load model.state_dict by now.')

        # NOTE load train mode state -> convert to deploy mode
        self.cfg.model.load_state_dict(state)
        self.model = self.cfg.model.deploy()
        self.postprocessor = self.cfg.postprocessor.deploy()
        # print(self.postprocessor.deploy_mode)

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)

def get_argparser():
    root_path = Path(r"D:/Workspace/Organoid_Tracking")
    # root_path = Path(r"/home/ubuntu/emma_myers")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default=root_path / "organoid_tracking/rtdetrv2_organoid/configs/rtdetrv2/rtdetrv2_r50vd_organoid.yml",
                        help="配置文件路径")
    parser.add_argument("--ckpt",
                        default=root_path / "organoid_tracking/rtdetrv2_organoid/output/exp_train/rtdetrv2_r50vd_organoid_stomach_cancer_epoch50_freeze_at_0/best.pth",
                        help="权重文件路径")
    parser.add_argument("--image_folder", 
                        default=root_path / "tracking_labeled/stomach_cancer_labeled/img_1",
                        help="待推理图片路径")
    parser.add_argument("--gt_file", default=root_path / "tracking_labeled/stomach_cancer_labeled/annotations/MOT/gt.txt", help="可选:GT文件路径，用于可视化")
    parser.add_argument('--output_dir', type=str, help='output directoy', default='./output/rtdetrv2_r101vd_6x_organoid')
    
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    # parser.add_argument("--track_high_thresh", type=float, default=1.0, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold valid for tracks")

    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=60, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--nms_thres", type=float, default=0.7, help='nms threshold')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="use Re-ID flag.")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')
    return parser

def main(args):
    results = []
    frame_id = 1
    output_dir = setup_experiment(args)

    # print(f'args.track_high_thresh:{args.track_high_thresh}\n')
    # print(f'args.track_low_thresh:{args.track_low_thresh}\n')
    device = torch.device(args.device)
    reader = ImageReader(orig_size=(3072, 2048), resize=(2400, 1600))
    # reader_2 = ImageReader(resize=(3072, 2048))
    model = Model(config=args.config, ckpt=args.ckpt).to(device)
    tracker = Deep_EIoU(args, frame_rate=30)  # 转换为字典参数
    gt_dict = load_gt_for_frame(args.gt_file, reader.orig_size, reader.resize) if args.gt_file else {}

    image_files = sorted(Path(args.image_folder).glob('*.jpg'))
    for img_path in tqdm(image_files, desc="Processing Images"):
        img_tensor = reader(img_path).to(device)
        with torch.no_grad():
            _, boxes, scores = model(img_tensor, torch.tensor([[img_tensor.shape[3], img_tensor.shape[2]]]).to(device))

        detections = torch.cat((boxes, scores.unsqueeze(-1)), 2).squeeze(0)
        tracked_objs = tracker.update(detections.cpu(), embedding=None)

        # 保存每一帧的tracking结果
        online_tlwhs, online_ids, online_scores = [], [], []
        for obj in tracked_objs:
            if obj.is_activated and obj.tlwh[2] * obj.tlwh[3] > 10:  # 面积过滤
                online_tlwhs.append(obj.tlwh)
                online_ids.append(obj.track_id)
                online_scores.append(obj.score)

        results.append((frame_id, online_tlwhs, online_ids, online_scores))
        frame_id += 1

        save_visualizations(reader.pil_img, detections.cpu().numpy(), tracked_objs, output_dir, img_path.name, vis_thresh=0.1, gt_boxes=gt_dict.get(frame_id, []))

    result_filename = output_dir / 'predict.txt'
    write_results(result_filename, results)


def _draw_boxes(draw, boxes, color_or_func, text_func=None, font_size=20):
    """通用绘制函数，支持 ndarray/对象两种情况，并支持画文字（置信度/ID）"""
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    for box in boxes:
        if isinstance(box, (list, tuple, np.ndarray)):
            coords = [(box[0], box[1]), (box[2], box[3])]
            draw.rectangle(coords, outline=color_or_func, width=3)
            if text_func:
                draw.text((box[0], box[1]-font_size), text_func(box), fill=color_or_func, font=font)
        else:
            bbox = box.tlbr
            coords = [(bbox[0], bbox[1]), (bbox[2], bbox[3])]
            color = color_or_func(box) if callable(color_or_func) else color_or_func
            draw.rectangle(coords, outline=color, width=3)
            if text_func:
                draw.text((bbox[0], bbox[1]-font_size), text_func(box), fill=color, font=font)


def save_visualizations(image, detections, tracked_objs, output_dir, filename, vis_thresh, gt_boxes=None):
    det_img = image.copy()
    det_boxes = [d[:5] for d in detections if d[4] > vis_thresh]
    _draw_boxes(ImageDraw.Draw(det_img), det_boxes, color_or_func='red', text_func=lambda x: f"{x[4]:.2f}", font_size=24)
    if gt_boxes:
        _draw_boxes(ImageDraw.Draw(det_img), gt_boxes, color_or_func='green', text_func=None, font_size=24)
    det_img.save(output_dir/"det"/filename)

    tra_img = image.copy()
    _draw_boxes(ImageDraw.Draw(tra_img), tracked_objs, color_or_func=lambda x: generate_colors(x.track_id), text_func=lambda x: f"ID:{x.track_id}", font_size=24)
    if gt_boxes:
        _draw_boxes(ImageDraw.Draw(tra_img), gt_boxes, color_or_func='green', text_func=None, font_size=24)
    tra_img.save(output_dir/"tra"/filename)

if __name__ == "__main__":
    args = get_argparser().parse_args()
    main(args)
    
