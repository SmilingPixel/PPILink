import json
import pathlib
from typing import List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


# dataset file schema
# {
#   "issues": [
#     {
#       "cluster": int,
#       "url": str,
#       "title": List[str],
#       "body": List[str],
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
#       "title": List[str],
#       "body": List[str],
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
        return len(self.links)

    def __getitem__(self, idx):
        link_info = self.links[idx]
        issue_idx, pr_idx, link = link_info['issue_idx'], link_info['pr_idx'], link_info['link']
        # TODO: consider more than title
        issue, pr = self.all_issues[issue_idx], self.all_prs[pr_idx]
        issue_nl, pr_nl = issue['title'], pr['title'] # both of their types are list of str (list of tokens)
        link_int = int(link) # 0 or 1

        issue_inputs = self.tokenizer(
            issue_nl,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_input_length,
            is_split_into_words=True, # https://huggingface.co/docs/transformers/v4.37.1/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__
        )
        pr_inputs = self.tokenizer(
            pr_nl,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_input_length,
            is_split_into_words=True, # https://huggingface.co/docs/transformers/v4.37.1/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__
        )
        for key in issue_inputs.keys():
            issue_inputs[key] = issue_inputs[key].squeeze(0)
        for key in pr_inputs.keys():
            pr_inputs[key] = pr_inputs[key].squeeze(0)
        return issue_inputs, pr_inputs, torch.tensor([link_int], dtype=torch.float32)
