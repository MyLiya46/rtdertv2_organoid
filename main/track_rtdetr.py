import os
import torch
import argparse
import warnings
import random
import motmetrics as mm
from pathlib import Path
from collections import OrderedDict
from loguru import logger
import glob

from src.utils import setup_logger
from tracker.byte_tracker import BYTETracker

def make_parser():
    parser = argparse.ArgumentParser("RT-DETR ByteTrack Tracker")
    parser.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    parser.add_argument("--nms", type=float, default=0.5, help="NMS threshold")
    parser.add_argument("--tsize", type=int, default=640, help="input image size")
    parser.add_argument("--track_thresh", type=float, default=0.5)
    parser.add_argument("--track_buffer", type=int, default=30)
    parser.add_argument("--match_thresh", type=float, default=0.8)
    parser.add_argument("--min_box_area", type=float, default=100)
    parser.add_argument("--save_result", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--ckpt", type=str, required=True, help="RT-DETR model checkpoint")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation images or video frames")
    parser.add_argument("--output_dir", type=str, default="./rtdetr_track_results")
    parser.add_argument("--image_folder", "tracking_labeled/all/img_1", help="待推理图片路径")
    return parser

def load_rtdetr_model(ckpt_path):
    model = torch.load(ckpt_path, map_location="cpu")["model"]
    model.eval().cuda()
    return model

def run_tracking(args):
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logger(args.output_dir, filename="track_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    # 加载模型
    model = load_rtdetr_model(args.ckpt)
    tracker = BYTETracker(
        args,
        frame_rate=30  # 你可以根据实际视频帧率调整
    )

    # 加载验证图像（这里只是一个例子，需你根据数据格式调整）
    image_paths = sorted(glob.glob(os.path.join(args.val_data, "*.jpg")))
    results = []

    for frame_id, img_path in enumerate(image_paths):
        img = torch.load(img_path)  # 或者用你自己的图像读取方式
        img_tensor = img.cuda().unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)[0]  # (num_det, 6) format

        if outputs is None or outputs.size(0) == 0:
            online_targets = []
        else:
            # 转换为 numpy
            output = outputs.detach().cpu().numpy()
            online_targets = tracker.update(output, img.shape[:2], img.shape[:2])

        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            x1, y1, w, h = tlwh
            results.append(
                [frame_id + 1, tid, x1, y1, w, h, -1, -1, -1]
            )

    result_path = os.path.join(args.output_dir, "results.txt")
    write_results(result_path, results, data_type="mot")

    logger.info("Tracking finished, results saved to {}".format(result_path))

    # -------- MOTA EVALUATION ----------
    gtfiles = glob.glob(os.path.join("datasets/mot/train", "*/gt/gt.txt"))
    tsfiles = [result_path]

    gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
    ts = OrderedDict([(os.path.splitext(Path(f).name)[0], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1)) for f in tsfiles])

    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gt:
            accs.append(mm.utils.compare_to_groundtruth(gt[k], tsacc, 'iou', distth=0.5))
            names.append(k)

    mh = mm.metrics.create()
    summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
    logger.info("\n" + mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))

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

if __name__ == "__main__":
    args = make_parser().parse_args()
    run_tracking(args)
