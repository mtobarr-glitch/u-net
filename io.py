from typing import Tuple
import os
import cv2
import numpy as np


def imread_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite_png(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if img.ndim == 3 and img.shape[2] == 3:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)
    else:
        cv2.imwrite(path, img)


def save_indexed_mask(path: str, mask: np.ndarray) -> None:
    imwrite_png(path, mask.astype(np.uint8))


def colorize_mask(mask: np.ndarray, colormap: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    cm = colormap
    for idx in range(len(cm)):
        color[mask == idx] = cm[idx]
    return color


def overlay_on_image(image_rgb: np.ndarray, mask_color_bgr: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    base = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(base, 1 - alpha, mask_color_bgr, alpha, 0)
    return overlay


def save_filament_binary(path: str, mask: np.ndarray, filament_class: int = 1) -> None:
    binm = (mask == filament_class).astype(np.uint8) * 255
    imwrite_png(path, binm)
