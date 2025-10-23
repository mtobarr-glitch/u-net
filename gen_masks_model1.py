import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

# Generate 8-bit indexed masks for Model 1: 0=background, 1=filaments, 2=flocs
# Heuristic rules:
# - Filaments: thin bright lines -> extract with high-frequency emphasis (Canny + thin dilation)
# - Flocs: larger bright blobs -> threshold after blur; remove filament pixels to avoid overlap


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def to_gray(rgb_bgr):
    if rgb_bgr.ndim == 3:
        return cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    return rgb_bgr


def extract_filaments(gray: np.ndarray) -> np.ndarray:
    # enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    # edge/line emphasis
    edges = cv2.Canny(g, 60, 150)
    # thin lines by small dilation then erosion to connect
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    fil = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=1)
    # optional thinning approximation:
    fil = cv2.dilate(fil, k, iterations=1)
    fil = cv2.erode(fil, k, iterations=2)
    fil = (fil > 0).astype(np.uint8)
    return fil


def extract_flocs(gray: np.ndarray, fil_mask: np.ndarray) -> np.ndarray:
    # smooth then threshold to get larger bright regions
    bl = cv2.GaussianBlur(gray, (7,7), 0)
    _, th = cv2.threshold(bl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # remove filaments from blobs
    th[fil_mask > 0] = 0
    # open/close to remove noise and fill holes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)
    # remove tiny components
    num, labels, stats, _ = cv2.connectedComponentsWithStats((th>0).astype(np.uint8), connectivity=8)
    area_min = 50
    fl = np.zeros_like(th, dtype=np.uint8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= area_min:
            fl[labels == i] = 1
    return fl


def build_mask(fil: np.ndarray, floc: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(fil, dtype=np.uint8)
    mask[floc > 0] = 2
    mask[fil > 0] = 1
    return mask


def main():
    ap = argparse.ArgumentParser(description='Generate 8-bit masks for Model 1 (0=bg,1=filaments,2=flocs)')
    ap.add_argument('--images', required=True, help='Input images directory (e.g., data/imgs)')
    ap.add_argument('--out', required=True, help='Output masks directory (e.g., data/masks_model1)')
    args = ap.parse_args()

    ensure_dir(args.out)
    names = sorted([f for f in os.listdir(args.images) if f.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff'))])
    for fn in tqdm(names, desc='gen masks'):
        in_path = os.path.join(args.images, fn)
        img = cv2.imread(in_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        g = to_gray(img)
        fil = extract_filaments(g)
        floc = extract_flocs(g, fil)
        mask = build_mask(fil, floc)
        base = os.path.splitext(fn)[0]
        out_path = os.path.join(args.out, base + '.png')
        cv2.imwrite(out_path, mask)

    print('Wrote masks to', args.out)


if __name__ == '__main__':
    main()
