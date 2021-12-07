#
#
#   Dataset
#
#

import torch
import pandas as pd

from torch.utils.data import Dataset
from transformers import BertTokenizer

from utils import md_to_text, calculate_pos_weights


class StackOverflowDataset(Dataset):

    def __init__(self, fpath):
        self._df = pd.read_csv(fpath, index_col="PostId")

        self._tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    @property
    def pos_weight(self):
        closed = self._df["OpenStatus"] != "open"
        pos_weight = calculate_pos_weights([closed.sum()], len(self._df))
        return pos_weight

    def __getitem__(self, index):
        row = self._df.iloc[index]

        title = row["Title"]
        body = md_to_text(row["BodyMarkdown"])
        text = title + ". " + body

        label = row["OpenStatus"] != "open"

        encoding = self._tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=180,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True
        )

        return {
            'text': text,
            'target': torch.tensor(label, dtype=torch.int32),
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

    def __len__(self):
        return len(self._df)
