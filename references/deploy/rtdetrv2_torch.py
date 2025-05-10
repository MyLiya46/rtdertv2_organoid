"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T

import numpy as np
from PIL import Image, ImageDraw

from src.core import YAMLConfig


def draw(images, labels, boxes, scores, thrh=0.6):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]

        for j, b in enumerate(box):
            draw.rectangle(list(b), outline='red', )
            draw.text((b[0], b[1]), text=f"{lab[j].item()} {round(scrs[j].item(), 2)}", fill='blue', )

        im.save(f'.\\results\\results_{i}.png')


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device)
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)

    im_pil = Image.open(args.im_file).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(args.device)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil)[None].to(args.device)

    output = model(im_data, orig_size)
    labels, boxes, scores = output

    draw([im_pil], labels, boxes, scores)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--config', type=str, )
    # parser.add_argument('-r', '--resume', type=str, )
    parser.add_argument('-c', '--config', type=str, default='configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml')
    parser.add_argument('-r', '--resume', type=str, default='checkpoints/rtdetrv2_r101vd_6x_coco_from_paddle.pth')
    # parser.add_argument('-f', '--im-file', type=str, )
    # parser.add_argument('-d', '--device', type=str, default='cuda:0')
    # parser.add_argument('--config', type=str,
                        # default='D:/Medical_segmentation/Kingmed/RT-DETR-main/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml')
    # parser.add_argument('--resume', type=str,
                        # default='D:/Medical_segmentation/Kingmed/RT-DETR-main/rtdetrv2_pytorch/tools/output/rtdetrv2_r50vd_6x_coco/best.pth')
    parser.add_argument('--im-file', type=str, default='D:/dataset/Intestinal Organoid Dataset/test/Subset_1_450x450_014.jpg')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    main(args)
