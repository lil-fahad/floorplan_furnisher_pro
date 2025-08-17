from pathlib import Path
from typing import List, Dict, Any
from ultralytics import YOLO
from PIL import Image

class Detector:
    """Wrapper around an Ultralytics YOLO model for inference."""

    def __init__(self, weights_path: str, device: str | None = None):
        w = Path(weights_path)
        if not w.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        self.model = YOLO(str(w))
        if device:
            self.model.to(device)
        self.names = self.model.model.names if hasattr(self.model.model, 'names') else self.model.names

    def predict(self, img: Image.Image, conf: float = 0.25) -> List[Dict[str, Any]]:
        """Run detection on a PIL image and return list of items."""
        res = self.model.predict(img, conf=conf, verbose=False)[0]
        boxes = res.boxes.xywh.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        scores = res.boxes.conf.cpu().numpy()
        items = []
        for (cx, cy, w, h), c, s in zip(boxes, clss, scores):
            name = self.names.get(int(c), str(c)) if isinstance(self.names, dict) else self.names[int(c)]
            items.append({
                "name": name,
                "cx": float(cx),
                "cy": float(cy),
                "w": float(w),
                "h": float(h),
                "confidence": float(s),
            })
        return items
