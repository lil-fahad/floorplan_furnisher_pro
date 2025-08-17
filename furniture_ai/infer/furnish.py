
from io import BytesIO
from typing import Dict, Any
import numpy as np
from PIL import Image, ImageDraw
from furniture_ai.parse.segmenter import Segmenter
from furniture_ai.parse.vectorizer import vectorize_floorplan
from furniture_ai.layout.generator import furnish_floorplan
from furniture_ai.config import settings

def run_furnish(image: Image.Image, weights_path: str | None = None) -> Dict[str, Any]:
    seg = Segmenter(weights_path=weights_path)
    mask = seg.predict(image).numpy()
    vec = vectorize_floorplan(mask)
    layout = furnish_floorplan(vec)
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
