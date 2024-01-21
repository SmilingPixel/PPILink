import json
import pathlib
from typing import List, Optional, Tuple, Union

import torch
import transformers
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


# dataset file schema
# {
#   "issues": [
#     {
#       "cluster": int,
#       "url": str,
#       "title": str,
#       "body": str,
#       "repo": str,
#       "created_at": str,
#       "closed_at": str or None
#     },
#     ...
#   ],
#   "prs": [
#     {
#       "cluster": int,
#       "url": str,
#       "title": str,
#       "body": str,
#       "repo": str,
#       "created_at": str,
#       "closed_at": str or None
#     },
#     ...
#   ],
#   "links": [
#     {
#       "issue_idx": int,
#       "pr_idx": int,
#       "link": bool
#     },
#     ...
#   ]
# }


class PILinkDataset(Dataset):
    """
    Dataset for PI Link

    """

    def __init__(
            self,
            file_path: Union[str, pathlib.Path],
            tokenizer: PreTrainedTokenizer,
            max_input_length: Optional[int] = None,
        ):
        super().__init__()

        # load dataset and process according to schema above
        with open(file_path, 'r') as f:
            raw_dataset = json.load(f)
        self.all_issues = raw_dataset['issues']
        self.all_prs = raw_dataset['prs']
        self.links = raw_dataset['links']

        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_input_length: Optional[int] = max_input_length

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        issue_idx, pr_idx, link = self.links[idx]
        # TODO: consider more than title
        issue, pr = self.all_issues[issue_idx], self.all_prs[pr_idx]
        issue_title, pr_title = issue['title'], pr['title']
        link_int = int(link) # 0 or 1

        issue_inputs = self.tokenizer(
            issue_title,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_input_length,
        )
        pr_inputs = self.tokenizer(
            pr_title,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_input_length,
        )
        return issue_inputs, pr_inputs, torch.tensor([link_int], dtype=torch.int8)
