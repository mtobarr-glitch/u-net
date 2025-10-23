import argparse
import os
import subprocess
import sys


def run(cmd, cwd=None):
    print("RUN:", " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    res.check_returncode()


def main():
    p = argparse.ArgumentParser(description='End-to-end pipeline for siteA model1: split, train, infer, quantify, eval')
    p.add_argument('--images', default='data/imgs')
    p.add_argument('--masks', default='data/masks_model1')
    p.add_argument('--split_out', default='data/splits/siteA')
    p.add_argument('--train_n', type=int, default=4)
    p.add_argument('--val_n', type=int, default=1)
    p.add_argument('--run_dir', default='runs/siteA_model1')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--early_stop', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--img_size', type=int, default=512)
    p.add_argument('--gen_masks', action='store_true', help='Generate masks from images before splitting')
    args = p.parse_args()

    # Optionally generate masks
    if args.gen_masks:
        run([sys.executable, 'scripts/gen_masks_model1.py', '--images', args.images, '--out', args.masks])

    # Prepare split
    run([sys.executable, 'scripts/prepare_split.py', '--images', args.images, '--masks', args.masks, '--out', args.split_out, '--train_n', str(args.train_n), '--val_n', str(args.val_n)])

    # Train
    run([sys.executable, 'scripts/train.py', '--data', args.split_out, '--labels', 'model1', '--epochs', str(args.epochs), '--early_stop', str(args.early_stop), '--save', args.run_dir, '--batch_size', str(args.batch_size), '--lr', str(args.lr), '--img_size', str(args.img_size)])

    # Infer on all original images
    weights = os.path.join(args.run_dir, 'best.pt')
    if not os.path.exists(weights):
        weights = os.path.join(args.run_dir, 'last.pt')
    preds_out = os.path.join(args.run_dir, 'preds')
    run([sys.executable, 'scripts/infer.py', '--weights', weights, '--images', args.images, '--out', preds_out, '--img_size', str(args.img_size), '--labels', 'model1'])

    # Quantify AF-UNet
    af_csv = os.path.join(args.run_dir, 'af_unet.csv')
    run([sys.executable, 'scripts/quantify_afunet.py', '--preds', preds_out, '--out', af_csv, '--img_area_mm2', '1.47', '--drop_vol_ml', '0.05'])

    # Evaluate vs masks
    metrics_csv = os.path.join(args.run_dir, 'metrics.csv')
    run([sys.executable, 'scripts/eval_metrics.py', '--preds', preds_out, '--gts', args.masks, '--report', metrics_csv])

    print('DONE')
    print('Artifacts:')
    print(' run_dir:', args.run_dir)
    print(' weights:', weights)
    print(' preds:', preds_out)
    print(' af_unet.csv:', af_csv)
    print(' metrics.csv:', metrics_csv)


if __name__ == '__main__':
    main()
