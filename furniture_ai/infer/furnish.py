
from io import BytesIO
from typing import Dict, Any
import numpy as np
from PIL import Image, ImageDraw
from furniture_ai.parse.segmenter import Segmenter
from furniture_ai.parse.detector import Detector
from furniture_ai.parse.vectorizer import vectorize_floorplan
from furniture_ai.layout.generator import furnish_floorplan
from furniture_ai.config import settings

def run_furnish(image: Image.Image, seg_weights: str | None = None, det_weights: str | None = None) -> Dict[str, Any]:
    seg = Segmenter(weights_path=seg_weights)
    mask = seg.predict(image).numpy()
    vec = vectorize_floorplan(mask)
    detections = []
    if det_weights:
        try:
            det = Detector(weights_path=det_weights)
            detections = det.predict(image)
        except FileNotFoundError:
            detections = []
    layout = furnish_floorplan(vec, detections)
    overlay = render_overlay(image, layout)
    return {"layout": layout, "overlay_png": overlay}

def render_overlay(image: Image.Image, layout: Dict[str, Any]) -> bytes:
    img = image.convert("RGBA").copy()
    d = ImageDraw.Draw(img, "RGBA")
    for r in layout.get("rooms", []):
        pts = [(float(x), float(y)) for (x,y) in r["polygon"]]
        d.polygon(pts, outline=(0,0,0,220))
        for it in r["items"]:
            cx, cy, w, h = it["cx"], it["cy"], it["w"], it["h"]
            rect = [cx-w/2, cy-h/2, cx+w/2, cy+h/2]
            d.rectangle(rect, outline=(255,0,0,220))
            d.text((cx, cy), it["name"], fill=(0,0,0,255), anchor="mm")
    bio = BytesIO(); img.convert("RGB").save(bio, format="PNG")
    return bio.getvalue()
