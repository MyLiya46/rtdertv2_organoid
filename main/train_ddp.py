"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os 
import sys
from pathlib import Path
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

from src.misc import dist_utils
from src.core import YAMLConfig, yaml_utils
from src.solver import TASKS


def main(args, ) -> None:
    """main
    """
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', ] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)
    
    # 构建最终输出路径
    image_dir = cfg.yaml_cfg.get('TrainDataset', {}).get('image_dir', 'unknown_dataset')
    dataset_name = Path(image_dir).parent.name if image_dir != 'unknown_dataset' else 'unknown_dataset'
    user_defined_output_dir = Path(args.output_dir)
    experiment_name = user_defined_output_dir.name
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    
    base_output_dir = Path("output/exp_train")
    exp_dir = base_output_dir / f"{experiment_name}_{dataset_name}_{timestamp}"
    args.output_dir = str(exp_dir)

    
    print(f'cfg: {cfg.__dict__}')
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # priority 0
    # parser.add_argument('-c', '--config', type=str, required=True, default='./configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml')
    # parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint', default='./configs/rtdetrv2_r20vd_6x_coco_ema.pth')
    # parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')
    # parser.add_argument('-d', '--device', type=str, help='device',)
    # parser.add_argument('--config', type=str, default='D:/Medical_segmentation/Kingmed/RT-DETR-main/rtdetrv2_organoid/configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml')
    # parser.add_argument('--resume', type=str, help='resume from checkpoint', default='D:/Medical_segmentation/Kingmed/RT-DETR-main/rtdetrv2_organoid/tools/output/rtdetrv2_r50vd_6x_coco_epoch150/best.pth')
    # parser.add_argument( '--resume', type=str, help='resume from checkpoint', default='D:/Medical_segmentation/Kingmed/RT-DETR-main/rtdetrv2_organoid/tools/output/rtdetrv2_r50vd_organoid_113/best.pth')
    parser.add_argument( '--config', type=str, default='configs/rtdetrv2/rtdetrv2_r101vd_6x_organoid.yml')
    parser.add_argument('--resume', type=str, help='resume from checkpoint', default=None)
    parser.add_argument('--tuning', type=str, help='tuning from checkpoint', default=None)
    parser.add_argument('--device', type=str, help='device', default='cuda:0')
    parser.add_argument('--seed', type=int, help='exp reproducibility', default=0)
    parser.add_argument('--use-amp', action='store_true', help='auto mixed precision training')

    parser.add_argument('--output_dir', type=str, help='output directoy', default='./output/rtdetrv2_r101vd_6x_organoid')
    # parser.add_argument('--output_dir', type=str, help='output directoy', default='output/rtdetrv2_organoid')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')
    parser.add_argument('--test-only', action='store_true', default=False,)

    # priority 1
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

    # env
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')
    parser.add_argument('--local-rank', type=int, help='local rank id')
    args = parser.parse_args()

    main(args)
