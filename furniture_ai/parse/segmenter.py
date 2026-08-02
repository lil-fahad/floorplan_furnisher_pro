from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models.segmentation import fcn_resnet50

from furniture_ai.config import settings
from furniture_ai.utils.logging import get_logger

log = get_logger("segmenter")


class Segmenter:
    def __init__(
        self,
        num_classes: int | None = None,
        weights_path: str | None = None,
        device: str | None = None,
        allow_untrained: bool | None = None,
    ) -> None:
        self.num_classes = num_classes or settings.segmenter.num_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = fcn_resnet50(
            weights=None,
            weights_backbone=None,
            num_classes=self.num_classes,
        )

        configured_path = weights_path or settings.segmenter.weights_path
        checkpoint = Path(configured_path).expanduser()
        permit_untrained = (
            settings.segmenter.allow_untrained if allow_untrained is None else allow_untrained
        )
        if checkpoint.is_file():
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            if not isinstance(state, dict):
                raise ValueError(f"Unsupported segmenter checkpoint format: {checkpoint}")
            self.model.load_state_dict(state, strict=True)
            log.info("Loaded segmenter weights from %s", checkpoint)
        elif not permit_untrained:
            raise FileNotFoundError(
                f"Segmenter weights not found at {checkpoint}. Train/download a checkpoint or set "
                "SEGMENTER__ALLOW_UNTRAINED=true only for development experiments."
            )
        else:
            log.warning("Running an untrained segmenter because allow_untrained is enabled")

        self.model.to(self.device).eval()
        self.pre = T.Compose(
            [
                T.Resize((settings.segmenter.img_size, settings.segmenter.img_size)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @torch.inference_mode()
    def predict(self, img: Image.Image) -> torch.Tensor:
        x = self.pre(img.convert("RGB")).unsqueeze(0).to(self.device)
        output = self.model(x)["out"]
        return torch.argmax(output, dim=1).squeeze(0).cpu()
