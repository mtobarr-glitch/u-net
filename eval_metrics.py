import argparse
import os
import glob
import cv2
import numpy as np
import pandas as pd
import sys

# Ensure repo root is on sys.path to import local packages when running this script directly
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sludge_seg.metrics.segmentation import confusion_per_class, metrics_from_confusion, metrics_to_dataframe, summarize_metrics


def load_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return m.astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description='Evaluate predicted masks against ground truth')
    p.add_argument('--preds', required=True)
    p.add_argument('--gts', required=True)
    p.add_argument('--report', required=True)
    args = p.parse_args()

    pred_files = sorted([f for f in glob.glob(os.path.join(args.preds, '*_mask.png'))])
    if not pred_files:
        pred_files = sorted(glob.glob(os.path.join(args.preds, '*.png')))

    all_stats = None
    ncls_max = 0

    for pf in pred_files:
        base = os.path.basename(pf)
        base_no_suffix = base.replace('_mask.png', '.png') if base.endswith('_mask.png') else base
        gt_path = os.path.join(args.gts, base_no_suffix)
        if not os.path.exists(gt_path):
            continue
        pred = load_mask(pf)
        gt = load_mask(gt_path)
        ncls = int(max(pred.max(), gt.max())) + 1
        ncls_max = max(ncls_max, ncls)
        stats = confusion_per_class(pred, gt, ncls)
        if all_stats is None:
            all_stats = {c: {k: 0 for k in ['tp','fp','fn','tn']} for c in range(ncls)}
        # ensure keys
        for c in range(ncls):
            if c not in all_stats:
                all_stats[c] = {k: 0 for k in ['tp','fp','fn','tn']}
        for c in range(ncls):
            for k in ['tp','fp','fn','tn']:
                all_stats[c][k] += stats[c][k]

    if all_stats is None:
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        pd.DataFrame([]).to_csv(args.report, index=False)
        print('No matching prediction/gt pairs found.')
        return

    mets = metrics_from_confusion(all_stats)
    class_names = {i: str(i) for i in range(ncls_max)}
    df = metrics_to_dataframe(mets, class_names)
    summary = summarize_metrics(mets)
    df_summary = pd.DataFrame([{**{'class_id': 'macro', 'class_name': 'macro'}, **summary}])
    out_df = pd.concat([df, df_summary], ignore_index=True)

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    out_df.to_csv(args.report, index=False)

    fil = mets.get(1, {})
    print('Filaments metrics:', fil)
    print('Report saved to', args.report)


if __name__ == '__main__':
    main()
