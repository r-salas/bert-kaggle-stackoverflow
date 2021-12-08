#
#
#   Dataset
#
#

import torch
import numpy as np
import pandas as pd
import multiprocessing as mp
import pytorch_lightning as pl

from typing import Optional
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from torch.utils.data import Dataset, DataLoader, Subset

from utils import md_to_text


class StackOverflowDataset(Dataset):

    def __init__(self, fpath, seed: Optional[int] = None):
        df = pd.read_csv(fpath, index_col="PostId")

        features = df.drop(columns=["OpenStatus"])

        label_encoder = LabelEncoder()
        targets = label_encoder.fit_transform(df["OpenStatus"])

        undersampler = RandomUnderSampler(random_state=seed)
        self.features, self.targets = undersampler.fit_resample(features, targets)

        self._tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    def __getitem__(self, index):
        features, target = self.features.iloc[index], self.targets[index]

        title = features["Title"]
        body = md_to_text(features["BodyMarkdown"])
        text = title + ". " + body

        encoding = self._tokenizer.encode_plus(
            text,
            max_length=428,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=True,
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'target': torch.tensor(target, dtype=torch.int32),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

    def __len__(self):
        return len(self.features)


class StackOverflowDataModule(pl.LightningDataModule):

    def __init__(self, fpath, batch_size: int = 16, num_workers: int = 0, seed: Optional[int] = 0):
        if num_workers < 0:
            num_workers = mp.cpu_count()

        self.seed = seed
        self.fpath = fpath
        self.batch_size = batch_size
        self.num_workers = num_workers

        self._train_indices = None
        self._val_indices = None
        self._test_indices = None

    @property
    def dataset(self):
        return StackOverflowDataset(self.fpath, self.seed)

    def setup(self, stage):
        indices = np.arange(len(self.dataset))

        train_indicess, valtest_indices = train_test_split(indices, test_size=0.3, random_state=self.seed)
        val_indices, test_indices = train_test_split(valtest_indices, test_size=0.5, random_state=self.seed)

        self._train_indices = train_indicess
        self._val_indices = val_indices
        self._test_indices = test_indices

    def train_dataloader(self):
        dataset = Subset(self.dataset, self._train_indices)
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return loader

    def val_dataloader(self):
        dataset = Subset(self.dataset, self._val_indices)
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return loader

    def test_dataloader(self):
        dataset = Subset(self.dataset, self._test_indices)
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return loader
