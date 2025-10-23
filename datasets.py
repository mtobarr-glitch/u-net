from typing import Optional, Tuple
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _resize_pad(img: np.ndarray, size: int = 512) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img_rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR if img.ndim == 3 else cv2.INTER_NEAREST)
    pad_h = size - nh
    pad_w = size - nw
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    if img.ndim == 3:
        img_pad = cv2.copyMakeBorder(img_rs, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
    else:
        img_pad = cv2.copyMakeBorder(img_rs, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)
    return img_pad


def _apply_clahe_rgb(img_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)


class SegDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: Optional[str], img_size: int = 512, clahe: bool = False, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.clahe = clahe
        self.transform = transform
        self.ids = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])

    def __len__(self):
        return len(self.ids)

    def _read_image(self, path: str) -> np.ndarray:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.clahe:
            rgb = _apply_clahe_rgb(rgb)
        return rgb

    def _read_mask(self, path: str) -> np.ndarray:
        m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if m is None:
            raise FileNotFoundError(path)
        if m.ndim == 3:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        return m.astype(np.uint8)

    def __getitem__(self, idx: int):
        name = self.ids[idx]
        img_path = os.path.join(self.images_dir, name)
        img = self._read_image(img_path)
        mask = None
        if self.masks_dir is not None:
            mask_path = os.path.join(self.masks_dir, os.path.splitext(name)[0] + '.png')
            mask = self._read_mask(mask_path)
        # resize+pad
        img = _resize_pad(img, self.img_size)
        if mask is not None:
            mask = _resize_pad(mask, self.img_size)
        # transforms
        if self.transform is not None:
            if mask is not None:
                aug = self.transform(image=img, mask=mask)
                img, mask = aug['image'], aug['mask']
            else:
                aug = self.transform(image=img)
                img = aug['image']
        # to tensor
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))
        img_t = torch.from_numpy(img)
        if mask is None:
            return img_t, name
        mask_t = torch.from_numpy(mask.astype(np.int64))
        return img_t, mask_t
