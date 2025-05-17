"""
多目标跟踪性能评估
使用MOTMetrics计算核心指标（MOTA/MOTP）
支持自定义IOU阈值（0.5）
输出格式化评估报告
兼容MOT Challenge数据格式
"""
import glob
import os
from collections import OrderedDict

from loguru import logger
import motmetrics as mm
from pathlib import Path


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.7))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names


def main():
    # evaluate MOTA
    mm.lap.default_solver = 'lap'

    gt_type = '_val_half'
    # gt_type = ''
    # print('gt_type', gt_type)
    # gt_files = glob.glob(
    #     os.path.join('datasets/mot/train', '*/gt/gt{}.txt'.format(gt_type)))
    # print('gt_files', gt_files)
    # pred_files = [f for f in glob.glob(os.path.join(results_folder, '*.txt')) if not os.path.basename(f).startswith('eval')]
    gt_files = glob.glob(r"D:\Workspace\Organoid_Tracking\tracking_labeled\stomach_cancer_labeled\annotations\MOT\gt.txt")
    pred_files = glob.glob(r"D:\Workspace\Organoid_Tracking\organoid_tracking\rtdetrv2_organoid\output\exp_track\rtdetrv2_r50vd_organoid_epoch50_freeze3stage_20250516-162630\predict.txt")

    logger.info('Found {} groundtruths and {} test files.'.format(len(gt_files), len(pred_files)))
    logger.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
    logger.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
    logger.info('Loading files.')

    '''file storage:yourdata/
    --video1/gt/gt.txt          # video1的gt文件
    --video2/gt/gt.txt          # video2的gt文件
    --video1.txt                # video1的test文件
    --video2.txt                # video2的test文件
    '''
    # gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1)) for f in gt_files])
    # ts = OrderedDict(
    #     [(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1)) for f in
    #      pred_files])
    '''file storage:img216/annotations/MOT/
    --gt.txt          # gt文件
    --predict.txt      # test文件
    '''
    # gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
    # ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1)) for f in tsfiles])
    gt = OrderedDict([('predict', mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1)) for f in gt_files])
    ts = OrderedDict([('predict', mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1)) for f in pred_files])

    mh = mm.metrics.create()
    accs, names = compare_dataframes(gt, ts)

    logger.info('Running metrics')
    metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked',
               'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
               'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    # summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
    # print(mm.io.render_summary(
    #   summary, formatters=mh.formatters,
    #   namemap=mm.io.motchallenge_metric_names))
    div_dict = {
        'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
        'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']}
    for divisor in div_dict:
        for divided in div_dict[divisor]:
            summary[divided] = (summary[divided] / summary[divisor])
    fmt = mh.formatters
    change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mostly_tracked',
                       'partially_tracked', 'mostly_lost']
    for k in change_fmt_list:
        fmt[k] = fmt['mota']
    print(mm.io.render_summary(summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names))

    metrics = mm.metrics.motchallenge_metrics + ['num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    logger.info('Completed')


if __name__ == "__main__":
    main()
