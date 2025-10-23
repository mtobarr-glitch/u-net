import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import sys

# Ensure repo root is on sys.path to import local packages when running this script directly
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sludge_seg.models.unet import UNet
from sludge_seg.config import get_colormap, num_classes
from sludge_seg_datamod.datasets import SegDataset
from sludge_seg_datamod.transforms import build_transforms
from sludge_seg.utils.io import imread_rgb, imwrite_png, colorize_mask, overlay_on_image, save_filament_binary


def main():
    p = argparse.ArgumentParser(description='Inference for U-Net segmentation')
    p.add_argument('--weights', required=True)
    p.add_argument('--images', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--img_size', type=int, default=512)
    p.add_argument('--labels', choices=['model1','model2'], default='model2')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ncls = num_classes(args.labels)
    model = UNet(in_channels=3, num_classes=ncls)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    tf = build_transforms('val')
    ds = SegDataset(args.images, masks_dir=None, img_size=args.img_size, clahe=False, transform=tf)

    cm = get_colormap(args.labels)

    for item in tqdm(ds.ids, desc='infer'):
        img_path = os.path.join(args.images, item)
        img_rgb = imread_rgb(img_path)
        # dataset preprocessing
        import cv2
        from sludge_seg_datamod.datasets import _resize_pad
        img_proc = _resize_pad(img_rgb, args.img_size)
        img = img_proc.astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))
        x = torch.from_numpy(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
        # save outputs
        base = os.path.splitext(item)[0]
        mask_path = os.path.join(args.out, f'{base}_mask.png')
        overlay_path = os.path.join(args.out, f'{base}_overlay.png')
        filbin_path = os.path.join(args.out, f'{base}_filaments_bin.png')
        imwrite_png(mask_path, pred.astype(np.uint8))
        color = colorize_mask(pred, cm)
        over = overlay_on_image(img_proc, color)
        imwrite_png(overlay_path, over)
        save_filament_binary(filbin_path, pred, filament_class=1)


if __name__ == '__main__':
    main()
