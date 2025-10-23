import argparse
import os
import glob
import numpy as np
import pandas as pd
import cv2
import sys

# Ensure repo root is on sys.path to import local packages when running this script directly
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sludge_seg.utils.postproc import area_fractions, af_unet_from_ra
from sludge_seg.config import IMG_AREA_MM2, DROP_VOL_ML


def is_mask_file(f: str) -> bool:
    return f.lower().endswith('.png') and ('_mask' in os.path.basename(f) or True)


essential_cols = ['image','af_unet_mm2_per_ml']


def main():
    p = argparse.ArgumentParser(description='Compute RA_i and AF-UNet from predicted masks')
    p.add_argument('--preds', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--img_area_mm2', type=float, default=IMG_AREA_MM2)
    p.add_argument('--drop_vol_ml', type=float, default=DROP_VOL_ML)
    args = p.parse_args()

    files = sorted([f for f in glob.glob(os.path.join(args.preds, '*.png')) if is_mask_file(f) and f.endswith('_mask.png')])
    if not files:
        files = sorted(glob.glob(os.path.join(args.preds, '*.png')))

    rows = []
    for f in files:
        mask = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ncls = int(mask.max()) + 1
        ra = area_fractions(mask, ncls)
        ra_fil = ra.get(1, 0.0)
        af = af_unet_from_ra(ra_fil, img_area_mm2=args.img_area_mm2, drop_vol_ml=args.drop_vol_ml)
        row = {'image': os.path.basename(f), 'af_unet_mm2_per_ml': af}
        for c in range(ncls):
            row[f'RA_{c}'] = ra[c]
        rows.append(row)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f'Wrote {args.out} with {len(rows)} rows')


if __name__ == '__main__':
    main()
