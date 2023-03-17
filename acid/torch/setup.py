from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import Trainer
from torchvision import transforms

from .. import conf
from ..contextmanagers import posix_path_compatibility
from ..decorators import classproperty
from .lightning_modules import BoardSegmentationModule, SquareClassifierModule

_lightning_trainer = None


def get_trainer():
    global _lightning_trainer
    if not _lightning_trainer:
        _lightning_trainer = Trainer(
            accelerator="auto", enable_progress_bar=False, enable_checkpointing=False, logger=False
        )
    return _lightning_trainer


class BoardModelSetup:
    image_size = (480, 300)
    dataset_path = Path(conf.TRAINING_DATA_DIR) / "boards"
    model_path = conf.DATA_DIR / "models" / "board_segmentation_mask.ckpt"

    _model = None

    @classproperty
    def model(cls):
        if not cls._model:
            with posix_path_compatibility():
                cls._model = BoardSegmentationModule.load_from_checkpoint(cls.model_path)
        return cls._model

    @classproperty
    def transforms(cls):
        return {
            "train": A.Compose(
                [
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.GridDistortion(p=0.5),
                    A.CLAHE(p=0.8),
                    A.RandomBrightnessContrast(p=0.8),
                    A.RandomGamma(p=0.8),
                    A.Normalize(),
                    A.Resize(cls.image_size[1], cls.image_size[0]),
                    ToTensorV2(),
                ]
            ),
            "val": A.Compose(
                [
                    A.Normalize(),
                    ToTensorV2(),
                ]
            ),
        }


class SquareModelSetup:
    image_size = (80, 80)
    image_normalize = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    dataset_path = Path(conf.TRAINING_DATA_DIR) / "squares" / "labeled"
    model_path = conf.DATA_DIR / "models" / "square_classifier.ckpt"
    num_classes = 3
    classes = ("black", "empty", "white")

    _model = None

    @classproperty
    def model(cls):
        if not cls._model:
            with posix_path_compatibility():
                cls._model = SquareClassifierModule.load_from_checkpoint(cls.model_path)
        return cls._model

    @classproperty
    def transforms(cls):
        means, stds = cls.image_normalize
        return {
            "train": transforms.Compose(
                [
                    transforms.RandomAdjustSharpness(0.25),
                    transforms.RandomAutocontrast(),
                    transforms.RandomEqualize(),
                    transforms.RandomRotation(30),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomAdjustSharpness(0.5),
                    transforms.Resize(100),
                    transforms.CenterCrop(95),
                    transforms.Resize(cls.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(means, stds),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(100),
                    transforms.CenterCrop(95),
                    transforms.Resize(cls.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(means, stds),
                ]
            ),
        }
