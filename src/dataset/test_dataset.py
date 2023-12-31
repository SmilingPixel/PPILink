from typing import List, Tuple

import torch
import transformers
from torch.utils.data import Dataset


class TestDataset(Dataset):
    def __init__(self, raw_dataset: List[Tuple[str, int]], tokenizer: transformers.PreTrainedTokenizer):
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        text, label = self.raw_dataset[idx]
        inputs: torch.Tensor = self.tokenizer(text, return_tensors='pt')
        for key in inputs:
            inputs[key] = inputs[key].squeeze(0)
        return inputs, torch.tensor([float(label)], dtype=torch.int8)
