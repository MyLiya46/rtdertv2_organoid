import argparse
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import motmetrics as mm
from loguru import logger

# MOTChallenge 格式最多 9 列
_COLS = ['frame', 'id', 'x', 'y', 'w', 'h', 'score', 'class', 'visibility']

def iou(boxA, boxB):
    xa, ya, wa, ha = boxA
    xb, yb, wb, hb = boxB
    xI = max(0, min(xa+wa, xb+wb) - max(xa, xb))
    yI = max(0, min(ya+ha, yb+hb) - max(ya, yb))
    inter = xI * yI
    union = wa*ha + wb*hb - inter
    return inter/union if union > 0 else 0

def remap_ids(gt_df, pred_df):
    """
    基于 IOU 构造 cost，匈牙利求解最优 pred_id → gt_id 映射
    """
    pred_ids = sorted(pred_df['id'].unique())
    gt_ids   = sorted(gt_df['id'].unique())
    cost = np.zeros((len(pred_ids), len(gt_ids)), dtype=float)
    for f in pred_df['frame'].unique():
        gsub = gt_df[gt_df['frame'] == f]
        psub = pred_df[pred_df['frame'] == f]
        for _, prow in psub.iterrows():
            for _, grow in gsub.iterrows():
                i = pred_ids.index(prow.id)
                j = gt_ids.index(grow.id)
                cost[i, j] -= iou((prow.x, prow.y, prow.w, prow.h),
                                  (grow.x, grow.y, grow.w, grow.h))
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {pred_ids[r]: gt_ids[c] for r, c in zip(row_ind, col_ind)}
    pred_df['id'] = pred_df['id'].map(mapping).fillna(pred_df['id']).astype(int)
    return pred_df

def load_mot_txt(path):
    """
    用 pandas 读入 MOT txt，自动判断列数。
    若列数 <6 报错；否则截取前 9 列并赋名
    """
    df = pd.read_csv(path, header=None)
    ncol = df.shape[1]
    if ncol < 6:
        raise ValueError(f"文件 {path} 列数 {ncol} 少于 MOT 支持的最少 6 列")
    df = df.iloc[:, :9]
    df.columns = _COLS[:df.shape[1]]
    return df

def to_mm_format(df):
    """
    将 DataFrame 转为 MOTMetrics 要求的格式：
    - index 为 MultiIndex (FrameId, Id)
    - 列名：X, Y, Width, Height, Confidence (可选)
    """
    mm_df = df[['frame', 'id', 'x', 'y', 'w', 'h']].copy()
    mm_df = mm_df.rename(columns={
        'frame': 'FrameId',
        'id': 'Id',
        'x': 'X', 'y': 'Y',
        'w': 'Width', 'h': 'Height'
    })
    if 'score' in df.columns:
        mm_df['Confidence'] = df['score'].values
    mm_df = mm_df.set_index(['FrameId', 'Id'])
    return mm_df

def compare_dataframes(gts, ts, iou_threshold=0.5):
    accs, names = [], []
    for seq in ts:
        if seq not in gts:
            logger.warning(f"No GT for {seq}, skipping.")
            continue
        logger.info(f"Comparing [{seq}] with IOU ≥ {iou_threshold}")
        acc = mm.utils.compare_to_groundtruth(
            gts[seq], ts[seq],
            dist='iou', distth=iou_threshold
        )
        accs.append(acc)
        names.append(seq)
    return accs, names

def main(args):
    mm.lap.default_solver = 'lap'
    iou_th = args.iou

    gt_path   = Path(args.gt)
    pred_path = Path(args.pred)
    seq_name  = gt_path.stem

    logger.info(f"Loading GT→{gt_path}  AND  Pred→{pred_path}")
    gt_df   = load_mot_txt(gt_path)
    pred_df = load_mot_txt(pred_path)

    logger.info("Remapping prediction IDs to GT IDs …")
    pred_df = remap_ids(gt_df, pred_df)

    # 转 mm 格式
    gts = OrderedDict([(seq_name, to_mm_format(gt_df))])
    ts  = OrderedDict([(seq_name, to_mm_format(pred_df))])

    mh = mm.metrics.create()
    accs, names = compare_dataframes(gts, ts, iou_threshold=iou_th)

    metrics = [
        'recall','precision','num_unique_objects',
        'mostly_tracked','partially_tracked','mostly_lost',
        'num_false_positives','num_misses',
        'num_switches','num_fragmentations',
        'mota','motp','num_objects'
    ]
    summary = mh.compute_many(accs, names=names,
                              metrics=metrics, generate_overall=True)

    # 归一化比率
    divs = {
        'num_objects': ['num_false_positives','num_misses','num_switches','num_fragmentations'],
        'num_unique_objects': ['mostly_tracked','partially_tracked','mostly_lost']
    }
    for div, keys in divs.items():
        for k in keys:
            summary[k] = summary[k] / summary[div]

    fmt = mh.formatters
    for k in [
        'num_false_positives','num_misses','num_switches','num_fragmentations',
        'mostly_tracked','partially_tracked','mostly_lost'
    ]:
        fmt[k] = fmt['mota']

    print(mm.io.render_summary(
        summary, formatters=fmt,
        namemap=mm.io.motchallenge_metric_names
    ))
    logger.info("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MOT eval w/ ID remapping & custom IOU threshold"
    )
    parser.add_argument('--gt',  type=str, required=True, help='GT txt 路径')
    parser.add_argument('--pred',type=str, required=True, help='预测 txt 路径')
    parser.add_argument('--iou', type=float, default=0.5,      help='IOU 阈值，默认0.5')
    args = parser.parse_args()
    main(args)