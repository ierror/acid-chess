import random
import tempfile
from pathlib import Path

import labelme2coco
import numpy as np
from albumentations.core.composition import Compose as ACompose
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            if isinstance(self.transform, ACompose):
                augmented = self.transform(image=x, mask=y)
                x, y = augmented["image"], augmented["mask"]
            else:
                x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


class COCOJsonDataset(Dataset):
    def __init__(
        self,
        data_path,
        split,
        image_size,
        transform=None,
    ):
        self.image_size = image_size
        self.transform = transform
        self.split = split
        self.data_path = data_path

        # convert labelme annotations to coco format and load img path + annotations
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as coco_json_tmp_dir:
            labelme2coco.convert(str(self.data_path), coco_json_tmp_dir)
            coco = COCO(Path(coco_json_tmp_dir) / "dataset.json")

        # we only use one annotation mask per img here...
        self.img_list = [img["file_name"] for img in coco.imgs.values()]
        self.mask_list = [coco.annToMask(anno) for anno in coco.anns.values()]

        # 80/20 split
        random_inst = random.Random(42)  # for repeatability
        n_items = len(self.img_list)
        idxs = random_inst.sample(range(n_items), n_items // 5)
        if self.split == "train":
            idxs = [idx for idx in range(n_items) if idx not in idxs]

        self.img_list = [self.img_list[i] for i in idxs]
        self.img_list = [Image.open(image) for image in self.img_list]
        self.img_list = [image.resize(self.image_size) for image in self.img_list]
        self.img_list = [np.array(image) for image in self.img_list]

        self.mask_list = [self.mask_list[i] for i in idxs]
        self.mask_list = [Image.fromarray(mask).convert("L") for mask in self.mask_list]
        self.mask_list = [mask.resize(self.image_size) for mask in self.mask_list]
        self.mask_list = [np.array(mask) for mask in self.mask_list]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = self.img_list[idx]
        mask = self.mask_list[idx]
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        return image, mask
