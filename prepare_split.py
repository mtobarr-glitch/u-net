import argparse
import os
import sys

# Ensure repo root is on sys.path to import local packages when running this script directly
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sludge_seg_datamod.split import create_split


def main():
    p = argparse.ArgumentParser(description="Build a reduced 20/20 (or custom) split from images and masks")
    p.add_argument('--images', required=True, help='Path to images dir')
    p.add_argument('--masks', required=True, help='Path to masks dir')
    p.add_argument('--out', required=True, help='Output split dir')
    p.add_argument('--train_n', type=int, default=20)
    p.add_argument('--val_n', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    create_split(args.images, args.masks, args.out, train_n=args.train_n, val_n=args.val_n, seed=args.seed)


if __name__ == '__main__':
    main()
