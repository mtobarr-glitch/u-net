# scripts/fuse_masks_model1.py
import argparse, os, glob
import numpy as np
import cv2

def read_bin(path, thresh=127, invert=False):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binm = (img > thresh).astype(np.uint8)
    return (1 - binm) if invert else binm

def main(args):
    os.makedirs(args.out, exist_ok=True)
    # Busca archivos tipo 1a_mask_fil.png  (ajusta si usas otra extensión)
    fil_paths = sorted(
        glob.glob(os.path.join(args.src, "*_mask_fil.png")) +
        glob.glob(os.path.join(args.src, "*_mask_fil.jpg")) +
        glob.glob(os.path.join(args.src, "*_mask_fil.jpeg"))
    )
    if not fil_paths:
        raise SystemExit(f"No encontré *_mask_fil.(png/jpg) en {args.src}")

    for pfil in fil_paths:
        ID = os.path.basename(pfil).split("_mask_fil")[0]
        # flóculos y (opcional) campo válido
        p_floc = None
        for ext in ("png","jpg","jpeg"):
            cand = os.path.join(args.src, f"{ID}_mask_floc.{ext}")
            if os.path.exists(cand): p_floc = cand; break
        p_bg = None
        for ext in ("png","jpg","jpeg"):
            cand = os.path.join(args.src, f"{ID}_mask_bg.{ext}")
            if os.path.exists(cand): p_bg = cand; break

        fil  = read_bin(pfil, thresh=args.thresh, invert=args.invert_fil)
        floc = read_bin(p_floc, thresh=args.thresh, invert=args.invert_floc) if p_floc else np.zeros_like(fil)
        valid= read_bin(p_bg,  thresh=args.thresh, invert=args.invert_bg)  if p_bg  else np.ones_like(fil)

        if floc.shape != fil.shape or valid.shape != fil.shape:
            raise ValueError(f"{ID}: tamaños de máscaras no coinciden.")

        # Fusión (prioridad filamento > flóculo > fondo) dentro del campo válido
        mask = np.zeros_like(fil, dtype=np.uint8)      # 0 fondo
        mask[(floc==1) & (valid==1)] = 2               # 2 flóculos
        mask[(fil==1)  & (valid==1)] = 1               # 1 filamentos (sobre-escribe)
        mask[valid==0] = 0

        outp = os.path.join(args.out, f"{ID}.png")
        cv2.imwrite(outp, mask)
        print("OK:", outp, "vals:", np.unique(mask))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="carpeta con *_mask_fil.*, *_mask_floc.* y opcional *_mask_bg.*")
    ap.add_argument("--out", required=True, help="carpeta de salida para máscaras indexadas 0/1/2")
    ap.add_argument("--thresh", type=int, default=127, help="umbral binarización 0..255")
    ap.add_argument("--invert_floc", action="store_true")
    ap.add_argument("--invert_fil", action="store_true")
    ap.add_argument("--invert_bg", action="store_true")
    args = ap.parse_args()
    main(args)
    
