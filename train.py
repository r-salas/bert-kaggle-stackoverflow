#
#
#   Train
#
#

import typer
import torch
import numpy as np
import pytorch_lightning as pl

from typing import Optional
from model import StackOverflowClassifier
from pytorch_lightning.loggers import WandbLogger
from dataset import StackOverflowDataModule, StackOverflowDataset


def train(data: str, output: str = "latest.ckpt", batch_size: int = 16, epochs: int = 10, num_workers: int = 0,
          seed: int = 0, gpus: int = -1, strategy: Optional[str] = None):
    if not torch.cuda.is_available():
        gpus = None
        strategy = None

    np.random.seed(seed)
    torch.manual_seed(seed)

    logger = WandbLogger(project="kaggle-stackoverflow")

    dataset = StackOverflowDataset(data, seed)
    model = StackOverflowClassifier(dataset.class_weight)
    datamodule = StackOverflowDataModule(data, batch_size=batch_size, num_workers=num_workers, seed=seed)

    trainer = pl.Trainer(gradient_clip_val=1.0, max_epochs=epochs, gpus=gpus, accelerator=strategy, logger=logger)

    trainer.fit(model, datamodule=datamodule)

    trainer.save_checkpoint(output)


if __name__ == "__main__":
    typer.run(train)
