import argparse
import os
import cv2
import numpy as np
from tqdm import trange

# Generate random thin curves (filament-like) and blobs

def draw_filaments(img, mask1, mask2):
    h, w = img.shape[:2]
    n_paths = np.random.randint(3, 8)
    for _ in range(n_paths):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        for _ in range(np.random.randint(30, 80)):
            dx, dy = np.random.randint(-5, 6), np.random.randint(-5, 6)
            x = np.clip(x + dx, 0, w-1)
            y = np.clip(y + dy, 0, h-1)
            cv2.circle(img, (x, y), 1, (255,255,255), -1)
            cv2.circle(mask1, (x, y), 1, 1, -1)  # class 1 filaments
            cv2.circle(mask2, (x, y), 1, 1, -1)


def draw_flocs(mask1, mask2):
    h, w = mask1.shape
    for _ in range(np.random.randint(3, 8)):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        r = np.random.randint(8, 25)
        cv2.circle(mask1, (x, y), r, 2, -1)  # class 2 flocs
        cv2.circle(mask2, (x, y), r, 2, -1)


def draw_shadows_artifacts(mask2):
    h, w = mask2.shape
    # shadows (3)
    for _ in range(np.random.randint(2, 6)):
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        x2, y2 = np.clip(x1 + np.random.randint(-50, 50), 0, w-1), np.clip(y1 + np.random.randint(-50, 50), 0, h-1)
        cv2.line(mask2, (x1, y1), (x2, y2), 3, thickness=np.random.randint(3,8))
    # artifacts (4)
    for _ in range(np.random.randint(5, 15)):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        cv2.rectangle(mask2, (x, y), (min(w-1, x+5), min(h-1, y+5)), 4, -1)
    # non_target (5)
    for _ in range(np.random.randint(2, 6)):
        pts = np.random.randint(0, min(w,h), (5,2))
        cv2.fillPoly(mask2, [pts], 5)


def main():
    p = argparse.ArgumentParser(description='Generate dummy sludge images and masks for smoke test')
    p.add_argument('--out', required=True)
    p.add_argument('--n', type=int, default=30)
    p.add_argument('--w', type=int, default=640)
    p.add_argument('--h', type=int, default=480)
    args = p.parse_args()

    img_dir = os.path.join(args.out, 'images')
    m1_dir = os.path.join(args.out, 'masks_model1')
    m2_dir = os.path.join(args.out, 'masks_model2')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(m1_dir, exist_ok=True)
    os.makedirs(m2_dir, exist_ok=True)

    for i in trange(args.n):
        img = np.zeros((args.h, args.w, 3), dtype=np.uint8)
        mask1 = np.zeros((args.h, args.w), dtype=np.uint8)
        mask2 = np.zeros((args.h, args.w), dtype=np.uint8)
        draw_filaments(img, mask1, mask2)
        draw_flocs(mask1, mask2)
        draw_shadows_artifacts(mask2)
        # save
        name = f'dummy_{i:03d}.png'
        cv2.imwrite(os.path.join(img_dir, name), img)
        cv2.imwrite(os.path.join(m1_dir, name), mask1)
        cv2.imwrite(os.path.join(m2_dir, name), mask2)

    print('Dummy data written to', args.out)


if __name__ == '__main__':
    main()
