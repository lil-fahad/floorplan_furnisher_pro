import argparse
import json
from pathlib import Path
import yaml
from ultralytics import YOLO

from furniture_ai.config import settings
from furniture_ai.utils.logging import get_logger

log = get_logger("train_detector")


def build_dataset_yaml(args, out_dir: Path) -> Path:
    ann_path = Path(args.train_ann)
    names = {}
    if ann_path.exists():
        with ann_path.open("r") as f:
            data = json.load(f)
            names = {c["id"]: c["name"] for c in data.get("categories", [])}
    yaml_path = out_dir / "dataset.yaml"
    with yaml_path.open("w") as f:
        yaml.safe_dump({
            "path": args.data_root,
            "train": args.train_ann,
            "val": args.val_ann,
            "names": names,
        }, f)
    return yaml_path


def parse_args() -> argparse.Namespace:
    cfg = settings.detector
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=cfg.data_root)
    p.add_argument("--train-ann", default=cfg.train_ann)
    p.add_argument("--val-ann", default=cfg.val_ann)
    p.add_argument("--img-size", type=int, default=cfg.img_size)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--lr", type=float, default=cfg.learning_rate)
    p.add_argument("--weights", help="Path to weights for resume/finetune", default=None)
    p.add_argument("--resume", action="store_true", help="Resume training from given weights")
    p.add_argument("--models-dir", default=cfg.models_dir)
    p.add_argument("--model", default=cfg.model)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = build_dataset_yaml(args, out_dir)
    model = YOLO(args.weights or args.model)
    log.info("Starting training")
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        lr0=args.lr,
        project=str(out_dir),
        name="detector",
        exist_ok=True,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
