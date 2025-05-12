"""
基于RT-DETR模型的PyTorch推理
"""

import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms
from PIL import Image, ImageDraw
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig
import argparse
from pathlib import Path
import time
from tqdm import tqdm
import datetime


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
    def __init__(self, confg=None, ckpt="") -> None:
        super().__init__()
        self.cfg = YAMLConfig(confg, resume=ckpt)
        if ckpt:
            checkpoint = torch.load(ckpt, map_location='cuda:0', weights_only=True)
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
    # root_path = Path(r"D:/Workspace/Organoid_Tracking")
    root_path = Path(r"/home/ubuntu/emma_myers")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default=root_path / "organoid_tracking/rtdetrv2_organoid/configs/rtdetrv2/rtdetrv2_r101vd_6x_organoid_linux.yml",
                        help="配置文件路径")
    parser.add_argument("--ckpt",
                        default=root_path / "organoid_tracking/rtdetrv2_organoid/output/exp_train_rtdetrv2_r101vd_6x_organoid_all_200epoch/best.pth",
                        help="权重文件路径")
    parser.add_argument("--image_folder", 
                        default=root_path / "tracking_labeled/stomach_cancer_labeled/img_1",
                        help="待推理图片路径")

    # parser.add_argument("--config", default="D:/Medical_segmentation/Kingmed/RT-DETR-main/rtdetrv2_organoid/configs/rtdetrv2/rtdetrv2_r50vd_organoid_113.yml", help="配置文件路径")
    # parser.add_argument("--ckpt", default="D:/Medical_segmentation/Kingmed/RT-DETR-main/rtdetrv2_organoid/tools/output/rtdetrv2_r50vd_organoid_113_test_16002400_epoch100/best.pth", help="权重文件路径")

    parser.add_argument("--output_dir", 
                        default=root_path / "organoid_tracking/rtdetrv2_organoid/output",
                        help="输出文件保存路径")
    parser.add_argument("--device", default="cuda:0")

    return parser


def main(args):
    # now = datetime.datetime.now()
    exp_name = f"exp_predict_{Path(args.ckpt).parent.stem}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        device = torch.device(f"cuda:{args.device.split(':')[1]}")
    else:
        device = torch.device("cpu")

    reader = ImageReader(orig_size=(3072, 2048), resize=(2400, 1600))
    model = Model(confg=args.config, ckpt=args.ckpt)
    model.to(device)

    img_paths = list(Path(args.image_folder).glob('*.jpg'))
    thrh = 0.1
    # thrh = 0.75
    for img_path in tqdm(img_paths, desc="Processing images", unit="image"):
        img = reader(img_path).to(device)
        _, _, H, W = img.shape
        size = torch.tensor([[W, H]]).to(device)
        output = model(img, size)
        labels, boxes, scores = output

        im = reader.pil_img
        draw = ImageDraw.Draw(im)
        for i in range(img.shape[0]):
            scr = scores[i]
            lab = labels[i][scr > thrh]
            box = boxes[i][scr > thrh]
            # print(f'scr={scr}, lab={lab}, box={box}')
            for b in box:
                draw.rectangle(list(b), outline='red')
                # draw.text((b[0], b[1]), text=str(scr[i]), fill='blue')

        # 保存图片到输出目录
        save_path = output_dir / img_path.name
        im.save(save_path)


if __name__ == "__main__":
    main(get_argparser().parse_args())
