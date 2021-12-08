#
#
#   Model
#
#

import torch
import numpy as np
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from typing import Optional
from sklearn.metrics import log_loss
from sklearn.metrics import ConfusionMatrixDisplay
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup


class StackOverflowClassifier(pl.LightningModule):

    def __init__(self, class_weights: Optional[np.ndarray] = None):
        super().__init__()

        self.save_hyperparameters()

        num_classes = 5

        if class_weights is None:
            class_weights = np.ones(num_classes)

        self.class_weights = class_weights

        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.drop = nn.Dropout(p=0.2)

        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size + 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        self._val_accuracy = torchmetrics.Accuracy(num_classes=num_classes)
        self._train_accuracy = torchmetrics.Accuracy(num_classes=num_classes)

    def forward(self, input_ids, attention_mask, meta):
        _, output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        output = torch.cat((meta, output), dim=1)
        return self.classifier(output)

    def training_step(self, batch, batch_idx):
        meta = batch["meta"]
        target = batch["target"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        y_pred = self(input_ids, attention_mask, meta)
        y_pred_proba = torch.softmax(y_pred, -1)

        loss = F.cross_entropy(y_pred, target.long(),
                               weight=torch.tensor(self.class_weights, dtype=torch.float32, device=self.device))

        self.log("train/loss", loss)

        self._train_accuracy(y_pred_proba, target)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5, correct_bias=False)

        dataset_size = len(self.trainer.datamodule.train_dataloader())
        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=max_estimated_steps
        )

        return [optimizer], [scheduler]

    def training_epoch_end(self, outputs):
        self.log_dict({
            "val/accuracy": self._train_accuracy.compute(),
        })
        self._train_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        meta = batch["meta"]
        target = batch["target"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        y_pred = self(input_ids, attention_mask, meta)
        y_pred_proba = torch.softmax(y_pred, -1)

        loss = F.cross_entropy(y_pred, target.long())

        if not self.trainer.sanity_checking:
            self.log("val/loss", loss)

            self._val_accuracy(y_pred_proba, target)

        return {"target": target.cpu().numpy(), "predictions": y_pred.cpu().numpy(),
                "probabilities": y_pred_proba.cpu().numpy()}

    def validation_epoch_end(self, outputs):
        if not self.trainer.sanity_checking:
            targets = np.concatenate([x["target"] for x in outputs])
            predictions = np.vstack([x["predictions"] for x in outputs])
            probabilities = np.vstack([x["probabilities"] for x in outputs])

            cm = ConfusionMatrixDisplay.from_predictions(targets, np.argmax(predictions, 1),
                                                         display_labels=["not a real question", "not constructive",
                                                                         "off topic", "open", "too localized"],
                                                         normalize="true", xticks_rotation="vertical")

            self.logger.experiment.log({
                "conf": cm.figure_
            })

            self.log_dict({
                "val/accuracy": self._val_accuracy.compute(),
                "val/log_loss": log_loss(targets, probabilities)
            })

        self._val_accuracy.reset()
