import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import sys

# Ensure repo root is on sys.path to import local packages when running this script directly
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sludge_seg.config import num_classes, LABEL_SCHEMAS
from sludge_seg.models.unet import UNet
from sludge_seg.losses.ce import build_ce_loss
from sludge_seg.losses.combined import CEDiceLoss
from sludge_seg_datamod.datasets import SegDataset
from sludge_seg_datamod.transforms import build_transforms
from sludge_seg.utils.train_utils import train_model, save_hparams
from sludge_seg.metrics.segmentation import confusion_per_class, metrics_from_confusion


def parse_args():
    p = argparse.ArgumentParser(description="Train U-Net for multi-class segmentation")
    p.add_argument('--data', required=True, help='Split directory with train/ and val/ subfolders')
    p.add_argument('--labels', choices=['model1','model2'], default='model2')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--early_stop', type=int, default=20)
    p.add_argument('--save', required=True, help='Run output directory')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--img_size', type=int, default=512)
    p.add_argument('--clahe', action='store_true')
    p.add_argument('--aug_shift', type=float, default=0.0)
    p.add_argument('--aug_zoom', type=float, default=0.0)
    p.add_argument('--aug_shear', type=float, default=0.0)
    p.add_argument('--aug_rot', type=float, default=0.0)
    p.add_argument('--aug_flip', action='store_true')
    p.add_argument('--loss', choices=['ce','ce_dice'], default='ce')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--site_mix', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    if args.seed:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ncls = num_classes(args.labels)
    model = UNet(in_channels=3, num_classes=ncls).to(device)

    if args.loss == 'ce':
        loss_fn = build_ce_loss()
    else:
        loss_fn = CEDiceLoss(alpha=0.5, beta=0.5)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # datasets
    train_dir = os.path.join(args.data, 'train')
    val_dir = os.path.join(args.data, 'val')

    tf_train = build_transforms('train', args.aug_shift, args.aug_zoom, args.aug_shear, args.aug_rot, args.aug_flip)
    tf_val = build_transforms('val')

    ds_train = SegDataset(os.path.join(train_dir, 'images'), os.path.join(train_dir, 'masks'), img_size=args.img_size, clahe=args.clahe, transform=tf_train)
    ds_val = SegDataset(os.path.join(val_dir, 'images'), os.path.join(val_dir, 'masks'), img_size=args.img_size, clahe=False, transform=tf_val)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0)

    os.makedirs(args.save, exist_ok=True)
    save_hparams(os.path.join(args.save, 'hparams.yaml'), vars(args))

    best_path, last_path, tr_losses, va_losses = train_model(
        model, optimizer, loss_fn, dl_train, dl_val, device, args.epochs, args.save, early_stop_patience=args.early_stop
    )

    # quick metrics report on val
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    import numpy as np
    from tqdm import tqdm
    all_stats = None
    with torch.no_grad():
        for imgs, masks in tqdm(dl_val, desc='metrics'):
            imgs = imgs.to(device)
            logits = model(imgs)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            gt = masks.numpy()
            for i in range(pred.shape[0]):
                stats = confusion_per_class(pred[i], gt[i], ncls)
                if all_stats is None:
                    all_stats = {c: {k: 0 for k in ['tp','fp','fn','tn']} for c in range(ncls)}
                for c in range(ncls):
                    for k in ['tp','fp','fn','tn']:
                        all_stats[c][k] += stats[c][k]
    mets = metrics_from_confusion(all_stats)
    # emphasize filaments (class id 1)
    fil = mets.get(1, {})
    with open(os.path.join(args.save, 'val_filaments_metrics.txt'), 'w') as f:
        f.write(str(fil))
    print('Filaments metrics (best weights):', fil)

    print('Saved:', best_path, last_path)


if __name__ == '__main__':
    main()
