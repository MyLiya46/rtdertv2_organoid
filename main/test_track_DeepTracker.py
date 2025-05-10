"""
检测+跟踪完整流程
RT-DETR检测+DeepTracker跟踪
多分辨率支持（检测2400x1600，输出3072x2048）
生成MOT格式的预测结果文件
可视化检测与跟踪结果（不同颜色区分）
"""
import torch
from torch import nn
from torchvision.transforms import transforms
from PIL import Image, ImageDraw
import sys

sys.path.append("..")
from src.core import YAMLConfig
import argparse
from pathlib import Path
import time
from tqdm import tqdm
from tracker.deep_tracker import DeepTracker
import colorsys
from PIL import ImageFont
import os


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},0,0,0\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id + 1, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    # print('save results to {}'.format(filename))


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


class ImageReader:
    def __init__(self, resize=(224, 224), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transform = transforms.Compose([
            # transforms.Resize((resize, resize)) if isinstance(resize, int) else transforms.Resize(
            #     (resize[0], resize[1])),
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean, std=std),
        ])
        self.resize = resize
        self.pil_img = None  # 保存最近一次读取的图片的pil对象

    def __call__(self, image_path, *args, **kwargs):
        """
        读取图片
        """
        self.pil_img = Image.open(image_path).convert('RGB').resize((self.resize[0], self.resize[1]))
        return self.transform(self.pil_img).unsqueeze(0)


class Model(nn.Module):
    def __init__(self, confg=None, ckpt="") -> None:
        super().__init__()
        self.cfg = YAMLConfig(confg, resume=ckpt)
        if ckpt:
            checkpoint = torch.load(ckpt, map_location='cpu')
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="D:/Medical_segmentation/Kingmed/RT-DETR-main/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_organoid_113.yml",
                        help="配置文件路径")
    parser.add_argument("--ckpt",
                        default="D:/Medical_segmentation/Kingmed/RT-DETR-main/rtdetrv2_pytorch/tools/output/rtdetrv2_r50vd_organoid_113_test_16002400_epoch100/best.pth",
                        help="权重文件路径")
    parser.add_argument("--image_folder", default="D:/dataset/organoids_tracking/pancreatic_cancer/Y202312006_P8/img/",
                        help="待推理图片路径")
    parser.add_argument("--output_dir_det",
                        default="D:/dataset/organoids_tracking/pancreatic_cancer/Y202312006_P8/img_visu/det/",
                        help="输出检测文件保存路径")
    parser.add_argument("--output_dir_tra",
                        default="D:/dataset/organoids_tracking/pancreatic_cancer/Y202312006_P8/img_visu/tra/",
                        help="输出跟踪文件保存路径")
    parser.add_argument("--output_dir_lab",
                        default="D:/dataset/organoids_tracking/pancreatic_cancer/Y202312006_P8/img_visu/label/",
                        help="输出跟踪文件保存路径")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument("--size_scale", type=float, default=3072/2400, help="the det model infers the default "
                                                                            "resolution at [2400,1600], "
                                                                            "for other resolution output need to fix,"
                                                                            "default [3072,2048]")

    # parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    # parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold valid for tracks")
    # parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    # parser.add_argument("--track_buffer", type=int, default=60, help="the frames for keep lost tracks")
    # parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    # parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    # parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    # parser.add_argument("--nms_thres", type=float, default=0.7, help='nms threshold')
    # parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    #
    # parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="use Re-ID flag.")
    # parser.add_argument('--proximity_thresh', type=float, default=0.5,
    #                     help='threshold for rejecting low overlap reid matches')
    # parser.add_argument('--appearance_thresh', type=float, default=0.25,
    #                     help='threshold for rejecting low appearance similarity reid matches')

    return parser


def main(args):
    device = torch.device(args.device)
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        device = torch.device(f"cuda:{args.device.split(':')[1]}")
    else:
        device = torch.device("cpu")

    reader = ImageReader(resize=(2400, 1600))
    reader_2 = ImageReader(resize=(3072, 2048))
    model = Model(confg=args.config, ckpt=args.ckpt)
    model.to(device=device)

    track = DeepTracker(args)
    # track = Deep_EIoU(args, frame_rate=30)

    Path(args.output_dir_det).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir_tra).mkdir(parents=True, exist_ok=True)

    img_paths = list(Path(args.image_folder).glob('*.jpg'))

    results = []
    frame_id = 1

    for img_path in tqdm(img_paths, desc="Processing images", unit="image"):
        img = reader(img_path).to(device)
        size = torch.tensor([[img.shape[3], img.shape[2]]]).to(device)
        start = time.time()
        output = model(img, size)
        labels, boxes, scores = output

        img_raw = reader_2(img_path).to(device)
        im = reader_2.pil_img
        draw = ImageDraw.Draw(im)
        thrh = 0.75

        # det visu draw
        for i in range(img.shape[0]):
            scr = scores[i]
            lab = labels[i][scr > thrh]
            box = boxes[i][scr > thrh]

            for b in box:
                draw.rectangle(list(b * args.size_scale), outline='red')
                draw.text((b[0] * args.size_scale, b[1] * args.size_scale), text=str(lab[i]), fill='blue')

        # save det visu results
        save_path_det = Path(args.output_dir_det) / img_path.name
        im.save(save_path_det)

        scores = scores.unsqueeze(-1)

        dets = torch.cat((boxes, scores), dim=2).squeeze(0)
        online_targets = track.update(output_results=dets, img_info=[1600, 2400], img_size=[1600, 2400])
        # online_targets = track.update(output_results=dets, embedding=None)

        online_tlwhs = []
        online_ids = []
        online_scores = []

        for t in online_targets:
            tlwh = t.tlwh * args.size_scale
            tid = t.track_id
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(t.score)
        # save label results
        results.append((frame_id, online_tlwhs, online_ids, online_scores))
        frame_id = frame_id + 1
        result_filename = os.path.join(args.output_dir_lab, 'predict.txt')
        write_results(result_filename, results)

        # tra visu draw
        img_raw = reader_2(img_path).to(device)
        im = reader_2.pil_img
        draw = ImageDraw.Draw(im)
        # thrh = 0.75

        for tracks in online_targets:
            # get bbox
            track_id = tracks.track_id
            tlwh = tracks.tlwh
            color = generate_colors(track_id)
            tlbr = [tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]]
            rectangle_coords = [(tlbr[0] * args.size_scale, tlbr[1] * args.size_scale), (tlbr[2] * args.size_scale, tlbr[3] * args.size_scale)]
            font = ImageFont.truetype("arial.ttf", 30)

            draw.rectangle(rectangle_coords, outline=color)
            draw.text((tlbr[0] * args.size_scale, tlbr[1] * args.size_scale), text=str(track_id), font=font, fill=color)

        # save tra visu results
        save_path_tra = Path(args.output_dir_tra) / img_path.name
        im.save(save_path_tra, dpi=(600, 600))


if __name__ == "__main__":
    main(get_argparser().parse_args())
