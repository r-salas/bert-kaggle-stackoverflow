#
#
#   Dataset
#
#

import torch
import pandas as pd

from typing import Optional
from torch.utils.data import Dataset
from transformers import BertTokenizer
from imblearn.under_sampling import RandomUnderSampler

from utils import md_to_text, calculate_pos_weights


class StackOverflowDataset(Dataset):

    def __init__(self, fpath, seed: Optional[int] = None):
        df = pd.read_csv(fpath, index_col="PostId")
        df["Closed"] = df["OpenStatus"] != "open"

        features = df.drop(columns=["Closed"])
        targets = df["Closed"]

        undersampler = RandomUnderSampler(random_state=seed)

        self.features, self.targets = undersampler.fit_resample(features, targets)

        self._tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    @property
    def pos_weight(self):
        pos_weight = calculate_pos_weights([self.targets.sum()], len(self.targets))
        return pos_weight

    def __getitem__(self, index):
        features, target = self.features.iloc[index], self.targets.iloc[index]

        title = features["Title"]
        body = md_to_text(features["BodyMarkdown"])
        text = title + ". " + body

        encoding = self._tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True
        )

        return {
            'text': text,
            'target': torch.tensor(target, dtype=torch.int32),
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

    def __len__(self):
        return len(self.features)
