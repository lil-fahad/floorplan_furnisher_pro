
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.segmentation import fcn_resnet50
from tqdm import tqdm

from furniture_ai.config import settings
from furniture_ai.utils.logging import get_logger
from furniture_ai.parse.dataset_seg import SegDataset

log = get_logger("train_segmenter")

def miou(pred, gt, num_classes):
    ious = []
    for c in range(num_classes):
        inter = np.logical_and(pred==c, gt==c).sum()
        union = np.logical_or(pred==c, gt==c).sum()
        if union>0: ious.append(inter/union)
    return float(np.mean(ious)) if ious else 0.0

def main():
    cfg = settings.segmenter
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = fcn_resnet50(num_classes=cfg.num_classes).to(device)
    train_ds = SegDataset(cfg.data_root, "train", img_size=cfg.img_size)
    val_ds   = SegDataset(cfg.data_root, "val", img_size=cfg.img_size)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    crit = nn.CrossEntropyLoss()
    best = 0.0
    out_dir = Path(cfg.models_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(cfg.epochs):
        model.train(); tr_loss=0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {ep+1}/{cfg.epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad()
            out = model(imgs)["out"]
            loss = crit(out, masks)
            loss.backward(); opt.step()
            tr_loss += float(loss.item())*imgs.size(0)
        tr_loss/=len(train_loader.dataset)

        # validate
        model.eval(); ious=[]
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                pred = model(imgs)["out"].argmax(1).cpu().numpy()
                gt   = masks.numpy()
                for p,g in zip(pred, gt): ious.append(miou(p,g,cfg.num_classes))
        val_miou = float(np.mean(ious)) if ious else 0.0
        log.info(f"ep={ep+1} loss={tr_loss:.4f} miou={val_miou:.4f}")
        if val_miou>best:
            best=val_miou; torch.save(model.state_dict(), out_dir/"best.pt"); log.info("saved best.pt")

if __name__=='__main__':
    main()
