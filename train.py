#
#
#   Train
#
#

import typer
import torch
import numpy as np
import torchmetrics
import torch.nn as nn
import multiprocessing as mp

from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from transformers import AdamW, get_linear_schedule_with_warmup

from model import StackOverflowClassifier
from dataset import StackOverflowDataset


def train(train_fpath, epochs: int = 10, device: str = "auto", num_workers: int = 0, seed: int = 0):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if num_workers < 0:
        num_workers = mp.cpu_count()

    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = StackOverflowDataset(train_fpath, seed)
    model = StackOverflowClassifier().to(device)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=num_workers)

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * epochs
    )

    criterion = nn.BCEWithLogitsLoss().to(device)

    pbar = tqdm(range(epochs))

    train_accuracy = torchmetrics.Accuracy(num_classes=1).to(device)
    train_precision = torchmetrics.Precision(num_classes=1).to(device)
    train_recall = torchmetrics.Recall(num_classes=1).to(device)

    val_accuracy = torchmetrics.Accuracy(num_classes=1).to(device)
    val_precision = torchmetrics.Precision(num_classes=1).to(device)
    val_recall = torchmetrics.Recall(num_classes=1).to(device)

    for epoch in tqdm(range(epochs)):
        cum_train_loss = 0.0

        train_accuracy.reset()
        train_precision.reset()
        train_recall.reset()

        for batch in tqdm(train_loader, leave=False, desc="Training"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["target"].to(device)

            y_pred = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).squeeze()
            y_pred_proba = torch.sigmoid(y_pred)

            loss = criterion(y_pred, target.float())

            optimizer.zero_grad()

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            cum_train_loss += loss.item()

            train_accuracy(y_pred_proba, target)
            train_precision(y_pred_proba, target)
            train_recall(y_pred_proba, target)

        cum_val_loss = 0.0

        val_accuracy.reset()
        val_precision.reset()
        val_recall.reset()

        for batch in tqdm(val_loader, leave=False, desc="Validating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["target"].to(device)

            with torch.no_grad():
                y_pred = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).squeeze()
                y_pred_proba = torch.sigmoid(y_pred)

                loss = criterion(y_pred, target.float())

            cum_val_loss += loss.item()

            val_accuracy(y_pred_proba, target)
            val_precision(y_pred_proba, target)
            val_recall(y_pred_proba, target)

        train_epoch_accuracy = train_accuracy.compute()
        train_epoch_recall = train_recall.compute()
        train_epoch_precision = train_precision.compute()
        train_epoch_loss = cum_train_loss / len(train_loader)

        val_epoch_accuracy = val_accuracy.compute()
        val_epoch_recall = val_recall.compute()
        val_epoch_precision = val_precision.compute()
        val_epoch_loss = cum_val_loss / len(val_loader)

        pbar.write(f"Epoch: {epoch} | Loss: {train_epoch_loss:.2f} | Val Loss: {val_epoch_loss:.2f} | "
                   f"Recall: {train_epoch_recall:.2f} | Val Recall: {val_epoch_recall:.2f} | "
                   f"Precision: {train_epoch_precision:.2f} | Val Precision: {val_epoch_precision:.2f} | "
                   f"Accuracy: {train_epoch_accuracy:.2f} | Val Accuracy: {val_epoch_accuracy:.2f}")


if __name__ == "__main__":
    typer.run(train)
