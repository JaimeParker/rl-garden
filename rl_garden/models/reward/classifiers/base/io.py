"""Shared IO helpers for classifier datasets."""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def decode_jpeg(data: np.ndarray) -> Optional[np.ndarray]:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode_jpeg_crop(
    data: np.ndarray,
    crop_region: Tuple[int, int, int, int],
) -> Optional[np.ndarray]:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    y0, y1, x0, x1 = crop_region
    cropped = img[y0:y1, x0:x1]
    return cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
