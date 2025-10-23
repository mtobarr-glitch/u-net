from typing import List
import os
import random
import shutil


def _ensure_dirs(path: str):
    os.makedirs(os.path.join(path, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(path, 'train', 'masks'), exist_ok=True)
    os.makedirs(os.path.join(path, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(path, 'val', 'masks'), exist_ok=True)


def create_split(images_dir: str, masks_dir: str, out_dir: str, train_n: int = 20, val_n: int = 20, seed: int = 42):
    random.seed(seed)
    ids = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    random.shuffle(ids)
    train_ids = ids[:train_n]
    val_ids = ids[train_n:train_n+val_n]
    _ensure_dirs(out_dir)
    for split, idlist in [('train', train_ids), ('val', val_ids)]:
        for name in idlist:
            src_img = os.path.join(images_dir, name)
            src_msk = os.path.join(masks_dir, os.path.splitext(name)[0] + '.png')
            dst_img = os.path.join(out_dir, split, 'images', name)
            dst_msk = os.path.join(out_dir, split, 'masks', os.path.splitext(name)[0] + '.png')
            shutil.copy2(src_img, dst_img)
            if os.path.exists(src_msk):
                shutil.copy2(src_msk, dst_msk)
