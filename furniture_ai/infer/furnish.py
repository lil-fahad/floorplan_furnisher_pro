from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from furniture_ai.config import settings
from furniture_ai.layout.constraints import rect_polygon
from furniture_ai.layout.generator import furnish_floorplan
from furniture_ai.parse.detector import Detector
from furniture_ai.parse.segmenter import Segmenter
from furniture_ai.parse.vectorizer import vectorize_floorplan


@lru_cache(maxsize=1)
def _segmenter() -> Segmenter:
    return Segmenter(weights_path=settings.segmenter.weights_path)


@lru_cache(maxsize=1)
def _detector() -> Detector | None:
    configured = settings.detector.weights_path
    if not configured:
        return None
    checkpoint = Path(configured).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Detector weights not found at {checkpoint}")
    return Detector(weights_path=str(checkpoint))


def run_furnish(image: Image.Image, *, pixels_per_cm: float | None = None) -> dict[str, Any]:
    rgb_image = image.convert("RGB")
    mask = _segmenter().predict(rgb_image).numpy().astype(np.uint8)
    if mask.shape != (rgb_image.height, rgb_image.width):
        mask = cv2.resize(
            mask,
            (rgb_image.width, rgb_image.height),
            interpolation=cv2.INTER_NEAREST,
        )
    vectorized = vectorize_floorplan(mask)

    detections: list[dict[str, Any]] = []
    detector = _detector()
    if detector is not None:
        detections = detector.predict(rgb_image)

    layout = furnish_floorplan(
        vectorized,
        detections,
        pixels_per_cm=pixels_per_cm,
    )
    overlay = render_overlay(rgb_image, layout)
    return {"layout": layout, "overlay_png": overlay}


def render_overlay(image: Image.Image, layout: dict[str, Any]) -> bytes:
    canvas = image.convert("RGBA").copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    for room in layout.get("rooms", []):
        room_points = [(float(x), float(y)) for x, y in room["polygon"]]
        draw.line(room_points + [room_points[0]], fill=(0, 0, 0, 220), width=2)
        for item in room["items"]:
            polygon = rect_polygon(
                float(item["cx"]),
                float(item["cy"]),
                float(item["w"]),
                float(item["h"]),
                float(item.get("angle", 0)),
            )
            points = [(float(x), float(y)) for x, y in polygon.exterior.coords]
            draw.polygon(points, fill=(255, 0, 0, 45), outline=(255, 0, 0, 220))
            draw.text(
                (float(item["cx"]), float(item["cy"])),
                str(item["name"]),
                fill=(0, 0, 0, 255),
                anchor="mm",
            )
    output = BytesIO()
    canvas.convert("RGB").save(output, format="PNG", optimize=True)
    return output.getvalue()
