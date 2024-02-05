import json
import pathlib
from typing import List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, RobertaTokenizer


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
            tokenizer: Union[BertTokenizer, RobertaTokenizer],
            max_input_length: int = 512,
        ):
        super().__init__()

        # load dataset and process according to schema above
        with open(file_path, 'r') as f:
            raw_dataset = json.load(f)
        self.all_issues = raw_dataset['issues']
        self.all_prs = raw_dataset['prs']
        self.links = raw_dataset['links']

        self.tokenizer: Union[BertTokenizer, RobertaTokenizer] = tokenizer
        self.max_input_length: int = max_input_length

    def __len__(self):
        return len(self.links)

    def __getitem__(self, idx):
        link_info = self.links[idx]
        issue_idx, pr_idx, link = link_info['issue_idx'], link_info['pr_idx'], link_info['link']
        issue, pr = self.all_issues[issue_idx], self.all_prs[pr_idx]
        issue_nl, pr_nl = issue['title'] + issue['body'], pr['title'] + pr['body'] # both of their types are list of str (list of tokens)

        # for some reason, we truncate the input separately by hand
        if self.tokenizer.__class__.__name__ == 'RobertaTokenizer':

            issue_nl_tokens: dict = self.tokenizer(
                issue_nl,
                return_tensors='pt',
                max_length=(self.max_input_length // 2),
                padding=False,
                truncation=True,
                is_split_into_words=True, # https://huggingface.co/docs/transformers/v4.37.1/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__
            )
            pr_nl_tokens: dict = self.tokenizer(
                pr_nl,
                return_tensors='pt',
                max_length=(self.max_input_length // 2), # fistr token <c> will be replaced with a <s> later
                padding=False,
                truncation=True,
                is_split_into_words=True,
            )

            # ATTENTION:
            # - RoBERTa doesn’t have token_type_ids (https://huggingface.co/docs/transformers/model_doc/roberta#overview)
            # - RoBERTa adds 2 <s> between sentences, see https://github.com/huggingface/transformers/issues/11519

            # make paddings
            padding_length = self.max_input_length - issue_nl_tokens['input_ids'].shape[1] - pr_nl_tokens['input_ids'].shape[1] + 1 # '+1' for <c> of pr_tokens
            paddings_tokens: dict = {
                'input_ids': torch.full((padding_length,), self.tokenizer.pad_token_id),
                'attention_mask': torch.full((padding_length,), 0),
            }

            # squeeze, remove <c> of second one, set token type ids of second one, and concat
            single_sep_tensor: torch.Tensor = torch.tensor([self.tokenizer.sep_token_id], dtype=torch.long)
            res_tokens: dict = {
                'input_ids': torch.cat(
                    (issue_nl_tokens['input_ids'][0], single_sep_tensor, pr_nl_tokens['input_ids'][0][1:], paddings_tokens['input_ids'])
                ), # [1:] to remove <c> of pr_nl_tokens, and add <s> before it
                'attention_mask': torch.cat(
                    (issue_nl_tokens['attention_mask'][0], pr_nl_tokens['attention_mask'][0], paddings_tokens['attention_mask'])
                ),
            }

        elif self.tokenizer.__class__.__name__ == 'BertTokenizer':

            # for some reason, we truncate the input separately by hand
            issue_tokens: dict = self.tokenizer(
                issue_nl,
                return_tensors='pt',
                max_length=(self.max_input_length // 2),
                padding=False,
                truncation=True,
                is_split_into_words=True, # https://huggingface.co/docs/transformers/v4.37.1/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__
            )
            pr_tokens: dict = self.tokenizer(
                pr_nl,
                return_tensors='pt',
                max_length=(self.max_input_length // 2 + 1), # fistr token [CLS] will be removed later
                padding=False,
                truncation=True,
                is_split_into_words=True,
            )

            # make paddings
            padding_length = self.max_input_length - issue_tokens['input_ids'].shape[1] - pr_tokens['input_ids'].shape[1] + 1 # '+1' for [CLS] of pr_tokens
            paddings_tokens: dict = {
                'input_ids': torch.full((padding_length,), self.tokenizer.pad_token_id),
                'attention_mask': torch.full((padding_length,), 0),
                'token_type_ids': torch.full((padding_length,), self.tokenizer.pad_token_type_id),
            }

            # squeeze, remove [CLS] of second one, set token type ids of second one, and concat
            res_tokens: dict = {
                'input_ids': torch.cat(
                    (issue_tokens['input_ids'][0], pr_tokens['input_ids'][0][1:], paddings_tokens['input_ids'])
                ), # [1:] to remove [CLS] of pr_tokens
                'attention_mask': torch.cat(
                    (issue_tokens['attention_mask'][0], pr_tokens['attention_mask'][0][1:], paddings_tokens['attention_mask'])
                ),
                'token_type_ids': torch.cat(
                    (issue_tokens['token_type_ids'][0], torch.ones_like(pr_tokens['token_type_ids'][0][1:]), paddings_tokens['token_type_ids'])
                ),
            }

        else:
            raise ValueError(f"Unsupported tokenizer: {self.tokenizer.__class__.__name__}")

        link_int = int(link) # False, True -> 0 or 1

        return res_tokens, torch.tensor([link_int], dtype=torch.float32)
