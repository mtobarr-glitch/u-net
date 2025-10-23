from typing import Dict
import numpy as np
from ..config import IMG_AREA_MM2, DROP_VOL_ML


def area_fractions(mask: np.ndarray, num_classes: int) -> Dict[int, float]:
    total = mask.size
    out = {}
    for c in range(num_classes):
        out[c] = float((mask == c).sum()) / max(total, 1)
    return out


def af_unet_from_ra(ra_filaments: float, img_area_mm2: float = IMG_AREA_MM2, drop_vol_ml: float = DROP_VOL_ML) -> float:
    return float((ra_filaments / drop_vol_ml) * img_area_mm2)
