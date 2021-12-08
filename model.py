#
#
#   Model
#
#

import torch
import wandb
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from transformers import BertModel, AdamW, get_linear_schedule_with_warmup


class StackOverflowClassifier(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.drop = nn.Dropout(p=0.5)
        self.out = nn.Linear(self.bert.config.hidden_size, 5)

        self._val_accuracy = torchmetrics.Accuracy(num_classes=5)
        self._train_accuracy = torchmetrics.Accuracy(num_classes=5)

    def forward(self, input_ids, attention_mask):
        _, output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        output = self.drop(output)
        output = self.out(output)
        return output

    def training_step(self, batch, batch_idx):
        target = batch["target"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        y_pred = self(input_ids, attention_mask).squeeze()
        y_pred_proba = torch.softmax(y_pred, -1)

        loss = F.cross_entropy(y_pred, target.long())

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
        target = batch["target"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        y_pred = self(input_ids, attention_mask)
        y_pred_proba = torch.softmax(y_pred, -1)

        loss = F.cross_entropy(y_pred, target.long())

        if not self.trainer.sanity_checking:
            self.log("val/loss", loss)

            self.logger.experiment.log({
                'conf': wandb.plot.confusion_matrix(
                    probs=y_pred_proba.cpu().numpy(), y_true=target.cpu().numpy(),
                    class_names=["not a real question", "not constructive", "off topic", "open", "too localized"],
                    title=f"Val: Epoch {self.current_epoch}"
                )
            })

            self._val_accuracy(y_pred_proba, target)

    def validation_epoch_end(self, outputs):
        self.log_dict({
            "val/accuracy": self._val_accuracy.compute(),
        })
        self._val_accuracy.reset()
