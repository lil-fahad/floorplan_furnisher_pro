
from typing import Dict, List
import numpy as np
import cv2
from shapely.geometry import Polygon

def _mask_to_polys(mask: np.ndarray) -> List[Polygon]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        if len(cnt) >= 3:
            pts = cnt.squeeze(1).astype(float)
            poly = Polygon(pts)
            if poly.is_valid and poly.area > 50.0:
                polys.append(poly)
    return polys

def vectorize_floorplan(mask: np.ndarray) -> Dict:
    mask_u8 = mask.astype(np.uint8)
    vec = {
        "walls": _mask_to_polys((mask_u8==1).astype(np.uint8)*255),
        "doors": _mask_to_polys((mask_u8==2).astype(np.uint8)*255),
        "windows": _mask_to_polys((mask_u8==3).astype(np.uint8)*255),
        "rooms": _mask_to_polys((mask_u8==4).astype(np.uint8)*255),
    }
    return vec
