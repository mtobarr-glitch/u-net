from typing import Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_transforms(phase: str = 'train',
                     aug_shift: float = 0.0,
                     aug_zoom: float = 0.0,
                     aug_shear: float = 0.0,
                     aug_rot: float = 0.0,
                     aug_flip: bool = False):
    tfs = []
    if phase == 'train':
        if aug_shift > 0 or aug_zoom > 0 or aug_rot:
            tfs.append(A.ShiftScaleRotate(shift_limit=aug_shift, scale_limit=aug_zoom, rotate_limit=aug_rot, border_mode=0, p=0.8))
        if aug_shear:
            tfs.append(A.Affine(shear=aug_shear, p=0.5))
        if aug_flip:
            tfs.extend([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)])
    return A.Compose(tfs)
