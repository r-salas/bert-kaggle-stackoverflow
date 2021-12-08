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
from dataset import StackOverflowDataModule


def train(fpath: str, batch_size: int = 16, epochs: int = 10, num_workers: int = 0, seed: int = 0, gpus: int = -1,
          accelerator: Optional[str] = "ddp"):
    if not torch.cuda.is_available():
        gpus = None
        accelerator = None

    np.random.seed(seed)
    torch.manual_seed(seed)

    model = StackOverflowClassifier()
    datamodule = StackOverflowDataModule(fpath, batch_size=batch_size, num_workers=num_workers, seed=seed)

    trainer = pl.Trainer(gradient_clip_val=1.0, max_epochs=epochs, gpus=gpus, accelerator=accelerator)

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    typer.run(train)
