from typing import Dict, List, Tuple
import numpy as np

# Constants
IMG_AREA_MM2: float = 1.47
DROP_VOL_ML: float = 0.05
DEFAULT_IMG_SIZE: int = 512

LABEL_SCHEMAS: Dict[str, Dict[str, int]] = {
    "model1": {  # 3 classes
        "background": 0,
        "filaments": 1,
        "flocs": 2,
    },
    "model2": {  # 6 classes
        "background": 0,
        "filaments": 1,
        "flocs": 2,
        "shadows": 3,
        "artifacts": 4,
        "non_target": 5,
    },
}

# Fixed color maps (BGR for OpenCV visualization)
COLORMAPS: Dict[str, List[Tuple[int, int, int]]] = {
    "model1": [
        (0, 0, 0),        # background (black)
        (0, 255, 255),    # filaments (yellow)
        (255, 0, 0),      # flocs (blue)
    ],
    "model2": [
        (0, 0, 0),        # background
        (0, 255, 255),    # filaments
        (255, 0, 0),      # flocs
        (128, 128, 128),  # shadows (gray)
        (0, 0, 255),      # artifacts (red)
        (0, 255, 0),      # non_target (green)
    ],
}


def num_classes(label_scheme: str) -> int:
    if label_scheme not in LABEL_SCHEMAS:
        raise ValueError(f"Unknown label scheme: {label_scheme}")
    return len(LABEL_SCHEMAS[label_scheme])


def get_colormap(label_scheme: str) -> np.ndarray:
    cm = COLORMAPS[label_scheme]
    return np.array(cm, dtype=np.uint8)
