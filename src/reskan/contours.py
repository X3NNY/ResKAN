from __future__ import annotations

import cv2
import numpy as np


def extract_contours(impedance: np.ndarray) -> np.ndarray:
    """
    Use Canny to extract contour features from impedance (2D model).
    Returns a binary {0,1} contour map.
    """
    norm_image = cv2.normalize(
        impedance, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    norm_image_to_255 = (norm_image * 255).astype(np.uint8)
    canny = cv2.Canny(norm_image_to_255, 10, 15)
    return np.clip(canny, 0, 1)

