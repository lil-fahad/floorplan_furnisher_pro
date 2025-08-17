
from pathlib import Path
import torch
from torchvision.models.segmentation import fcn_resnet50
import torchvision.transforms as T
from PIL import Image
from furniture_ai.config import settings
from furniture_ai.utils.logging import get_logger

log = get_logger("segmenter")

class Segmenter:
    def __init__(self, num_classes: int | None = None, weights_path: str | None = None, device: str | None = None):
        self.num_classes = num_classes or settings.segmenter.num_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = fcn_resnet50(num_classes=self.num_classes)
        self.model.to(self.device).eval()
        if weights_path and Path(weights_path).exists():
            state = torch.load(weights_path, map_location=self.device)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self.model.load_state_dict(state, strict=False)
            log.info(f"Loaded weights: {weights_path}")
        self.pre = T.Compose([T.Resize((settings.segmenter.img_size,settings.segmenter.img_size)),
                              T.ToTensor(),
                              T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

    @torch.inference_mode()
    def predict(self, img: Image.Image):
        x = self.pre(img.convert("RGB")).unsqueeze(0).to(self.device)
        out = self.model(x)["out"]  # (1,C,H,W)
        pred = torch.argmax(out, dim=1).squeeze(0).cpu()
        return pred
