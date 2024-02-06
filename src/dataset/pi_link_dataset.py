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
#       "filtered_commits": 
#       {
#         "url": str,
#         "message": str,
#         "files": 
#         [
#           {
#             "filename": str,
#             "additions": int,
#             "deletions": int,
#             "status": str,
#             "patch": str
#           }
#         ]
#       }
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

    Attributes:
        nlnl_model_tokenizer (Union[BertTokenizer, RobertaTokenizer]): The tokenizer for natural language to natural language model.
        nlpl_model_tokenizer (RobertaTokenizer): The tokenizer for natural language to program language model.
        max_input_length (int): The maximum input length for tokenization.
        all_issues (List[dict]): The list of all issues.
        all_prs (List[dict]): The list of all PRs.
        links (List[dict]): The list of links between issues and PRs (label is 0 or 1).

    """

    def __init__(
            self,
            file_path: Union[str, pathlib.Path],
            nlnl_model_tokenizer: Union[BertTokenizer, RobertaTokenizer],
            nlpl_model_tokenizer: RobertaTokenizer,
            max_input_length: int = 512,
        ):
        super().__init__()

        # load dataset and process according to schema above
        with open(file_path, 'r') as f:
            raw_dataset = json.load(f)
        self.all_issues = raw_dataset['issues']
        self.all_prs = raw_dataset['prs']
        self.links = raw_dataset['links']

        self.nlnl_model_tokenizer: Union[BertTokenizer, RobertaTokenizer] = nlnl_model_tokenizer
        self.nlpl_model_tokenizer: RobertaTokenizer = nlpl_model_tokenizer
        self.max_input_length: int = max_input_length

    def __len__(self):
        return len(self.links)

    def __getitem__(self, idx):
        """
        Get the item at the given index.

        Args:
            idx (int): The index of the item to get.
        
        Returns:
            Tuple[dict, dict, torch.Tensor]: The tokenized natural language to natural language pair, the tokenized natural language to program language pair, and the link label.
        
        """
        link_info = self.links[idx]
        issue_idx, pr_idx, link = link_info['issue_idx'], link_info['pr_idx'], link_info['link']
        issue, pr = self.all_issues[issue_idx], self.all_prs[pr_idx]
        issue_nl, pr_nl = issue['title'] + issue['body'], pr['title'] + pr['body'] # both of their types are list of str (list of tokens)

        nlnl_res_tokens = self.tokenize_nlnl_pair((issue_nl, pr_nl))
        

        # TODO: Consider more commits (We only consider the first file in the first commit for now)
        commits: List[dict] = pr['filtered_commits']
        pr_pl: str = ''
        for commit in commits:
            for file in commit['files']:
                if len(file['patch']) > 0:
                    pr_pl = file['patch']
                    break # TODO: delete
            if len(pr_pl) > 0:
                break

        nlpl_res_tokens = self.tokenize_nlpl_pair((issue_nl, pr_pl))
        
        link_int = int(link) # False, True -> 0 or 1

        return nlnl_res_tokens, nlpl_res_tokens, torch.tensor([link_int], dtype=torch.float32)
    
    def tokenize_nlnl_pair(self, nlnl_pair: Tuple[List[str], List[str]]) -> dict:
        """
        Tokenize a natural language pair.

        Args:
            nlnl_pair (Tuple[List[str], List[str]]): The natural language pair to tokenize.
                The first element is the issue natural language and the second element is the PR natural language.
        
        Returns:
            dict: The tokenized natural language pair.
        """

        issue_nl, pr_nl = nlnl_pair

        # for some reason, we truncate the input separately by hand
        if self.nlnl_model_tokenizer.__class__.__name__ == 'RobertaTokenizer':

            issue_nl_tokens: dict = self.nlnl_model_tokenizer(
                issue_nl,
                return_tensors='pt',
                max_length=(self.max_input_length // 2),
                padding=False,
                truncation=True,
                is_split_into_words=True, # https://huggingface.co/docs/transformers/v4.37.1/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__
            )
            pr_nl_tokens: dict = self.nlnl_model_tokenizer(
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
            padding_length = self.max_input_length - issue_nl_tokens['input_ids'].shape[1] - pr_nl_tokens['input_ids'].shape[1]
            paddings_tokens: dict = {
                'input_ids': torch.full((padding_length,), self.nlnl_model_tokenizer.pad_token_id),
                'attention_mask': torch.full((padding_length,), 0),
            }

            # squeeze, remove <c> of second one, and concat
            single_sep_tensor: torch.Tensor = torch.tensor([self.nlnl_model_tokenizer.sep_token_id], dtype=torch.long)
            nlnl_res_tokens: dict = {
                'input_ids': torch.cat(
                    (issue_nl_tokens['input_ids'][0], single_sep_tensor, pr_nl_tokens['input_ids'][0][1:], paddings_tokens['input_ids'])
                ), # [1:] to remove <c> of pr_nl_tokens, and add <s> before it
                'attention_mask': torch.cat(
                    (issue_nl_tokens['attention_mask'][0], pr_nl_tokens['attention_mask'][0], paddings_tokens['attention_mask'])
                ),
            }

            return nlnl_res_tokens

        elif self.nlnl_model_tokenizer.__class__.__name__ == 'BertTokenizer':

            issue_tokens: dict = self.nlnl_model_tokenizer(
                issue_nl,
                return_tensors='pt',
                max_length=(self.max_input_length // 2),
                padding=False,
                truncation=True,
                is_split_into_words=True, # https://huggingface.co/docs/transformers/v4.37.1/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__
            )
            pr_tokens: dict = self.nlnl_model_tokenizer(
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
                'input_ids': torch.full((padding_length,), self.nlnl_model_tokenizer.pad_token_id),
                'attention_mask': torch.full((padding_length,), 0),
                'token_type_ids': torch.full((padding_length,), self.nlnl_model_tokenizer.pad_token_type_id),
            }

            # squeeze, remove [CLS] of second one, set token type ids of second one, and concat
            nlnl_res_tokens: dict = {
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

            return nlnl_res_tokens

        else:
            raise ValueError(f"Unsupported tokenizer: {self.nlnl_model_tokenizer.__class__.__name__}")
        
    def tokenize_nlpl_pair(self, nlpl_pair: Tuple[List[str], str]) -> dict:
        """
        Tokenize a natural language and program language pair.

        Args:
            nlpl_pair (Tuple[List[str], str]): The natural language and program language pair to tokenize.
                The first element is the natural language and the second element is the program language.
                Note: The program language is a single string, not a list of strings(splited words).
        
        Returns:
            dict: The tokenized natural language and program language pair.
        """

        issue_nl, pr_pl = nlpl_pair

        # see [Li, Zhiyu, et al. "Automating code review activities by large-scale pre-training."
        # Proceedings of the 30th ACM Joint European Software Engineering Conference
        # and Symposium on the Foundations of Software Engineering. 2022.]

        # <s> [code tokens] <msg> [natural language tokens] <pad> (NO sep token)

        pr_nl_tokens: dict = self.nlpl_model_tokenizer(
            pr_pl,
            return_tensors='pt',
            max_length=(self.max_input_length // 2),
            padding=False,
            truncation=True,
            is_split_into_words=False,
        )

        issue_nl_tokens: dict = self.nlpl_model_tokenizer(
            issue_nl,
            return_tensors='pt',
            max_length=(self.max_input_length // 2), # fistr token <c> will be replaced with a <msg> later
            padding=False,
            truncation=True,
            is_split_into_words=True, # https://huggingface.co/docs/transformers/v4.37.1/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__
        )

        # ATTENTION:
        # - RoBERTa doesn’t have token_type_ids (https://huggingface.co/docs/transformers/model_doc/roberta#overview)

        # make paddings
        padding_length = self.max_input_length - (issue_nl_tokens['input_ids'].shape[1] - 1) - (pr_nl_tokens['input_ids'].shape[1] - 1)
        paddings_tokens: dict = {
            'input_ids': torch.full((padding_length,), self.nlpl_model_tokenizer.pad_token_id),
            'attention_mask': torch.full((padding_length,), 0),
        }

        # squeeze, remove <c> of second one, add <msg>, and concat
        single_msg_tensor: torch.Tensor = torch.tensor([self.nlpl_model_tokenizer.convert_tokens_to_ids('<msg>')], dtype=torch.long)
        nlpl_res_tokens: dict = {
            'input_ids': torch.cat(
                (issue_nl_tokens['input_ids'][0][:-1], single_msg_tensor, pr_nl_tokens['input_ids'][0][1:-1], paddings_tokens['input_ids'])
            ), # [1:] to remove <c> of pr_nl_tokens, and add <msg> before it
            'attention_mask': torch.cat(
                (issue_nl_tokens['attention_mask'][0][:-1], pr_nl_tokens['attention_mask'][0][:-1], paddings_tokens['attention_mask'])
            ),
        }

        return nlpl_res_tokens
