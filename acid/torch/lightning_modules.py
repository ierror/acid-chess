import torch
import torch.nn.functional as F
import torchvision
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy

from .models import BoardSegmenationUNet


class SquareClassifierModule(LightningModule):
    def __init__(self, train_loader, val_loader, test_loader, batch_size, lr=0.001, *kwargs):
        super().__init__()

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.batch_size = batch_size
        self.lr = lr

        self.save_hyperparameters()

        self.model = torchvision.models.resnet18(weights=None, num_classes=3)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model.maxpool = nn.Identity()

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, "multiclass", num_classes=3)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = len(self.train_loader) * 64
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


class BoardSegmentationModule(LightningModule):
    def __init__(self, batch_size, num_workers, train_set=None, val_set=None, lr=0.001, *kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.train_set = train_set
        self.val_set = val_set

        self.save_hyperparameters()

        self.net = BoardSegmenationUNet(
            num_classes=2,
            num_layers=5,
            bilinear=False,
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss = F.cross_entropy(out, mask)
        log_dict = {"train_loss": loss}
        self.log("train_loss", loss)
        return {"loss": loss, "log": log_dict, "progress_bar": log_dict}

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss_val = F.cross_entropy(out, mask)
        self.log("val_loss", loss_val)
        return {"val_loss": loss_val}

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x["val_loss"] for x in outputs]).mean()
        log_dict = {"val_loss": loss_val}
        self.log("val_loss", log_dict["val_loss"])
        return {"log": log_dict, "val_loss": log_dict["val_loss"], "progress_bar": log_dict}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
