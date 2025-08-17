
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class SegDataset(Dataset):
    def __init__(self, root: str, split: str = "train", img_size: int = 512):
        self.imgs = sorted((Path(root)/"images"/split).glob("*"))
        self.masks = sorted((Path(root)/"masks"/split).glob("*"))
        assert len(self.imgs) == len(self.masks), "Images/masks mismatch"
        self.tf_img = T.Compose([T.Resize((img_size,img_size)), T.ToTensor(),
                                 T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        self.img_size = img_size

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        im = Image.open(self.imgs[idx]).convert("RGB")
        m  = Image.open(self.masks[idx]).convert("L")  # single channel ids
        im = self.tf_img(im)
        m  = m.resize((self.img_size,self.img_size), Image.NEAREST)
        m  = torch.from_numpy(np.array(m, dtype=np.int64))
        return im, m
