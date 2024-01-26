from typing import Any, List

import torch.nn as nn
import torch
from transformers import BertModel, PreTrainedModel


class PILinkModel(nn.Module):
    def __init__(self, pr_code_model: Any, pr_nlp_model: BertModel, issue_nlp_model: BertModel):
        super(PILinkModel, self).__init__()
        self.pr_nlp_model: BertModel = pr_nlp_model
        self.issue_nlp_model: BertModel = issue_nlp_model
        # TODO: add code model

        total_hidden_size: int = pr_nlp_model.config.hidden_size + issue_nlp_model.config.hidden_size # + pr_code_model.config.hidden_size
        linear_sizes: List[int] = [total_hidden_size, 512, 1]
        self.linears = nn.Sequential(
            nn.Linear(linear_sizes[0], linear_sizes[1]),
            nn.ReLU(),
            nn.Linear(linear_sizes[1], linear_sizes[2]),
            nn.Sigmoid()
        )
      
    def forward(self, issue_nlp_inputs, pr_nlp_inputs):
        """
        This function takes as input two dictionaries, issue_nlp_inputs and pr_nlp_inputs, which are expected to contain 
        the input data for the issue and pull request (PR) NLP models respectively. These dictionaries are unpacked and 
        passed to their respective models.

        The output from each model is a tensor representing the last hidden state of the model. The first element of 
        each tensor (corresponding to the [CLS] token in BERT-based models) is selected and these two tensors are 
        concatenated along dimension 1.

        This concatenated tensor is then passed through a series of linear layers (self.linears) to produce the final 
        output.

        Parameters:
            issue_nlp_inputs (dict): The input data for the issue NLP model.
            pr_nlp_inputs (dict): The input data for the PR NLP model.

        Returns:
            torch.Tensor: The output from the series of linear layers.
        """
        issue_nlp_vec = self.issue_nlp_model(**issue_nlp_inputs).last_hidden_state
        issue_nlp_vec = issue_nlp_vec[:,0,:] # torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)
        pr_nlp_vec = self.pr_nlp_model(**pr_nlp_inputs).last_hidden_state
        pr_nlp_vec = pr_nlp_vec[:,0,:] # torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)
        vec = torch.cat((issue_nlp_vec, pr_nlp_vec), dim=1)
        out = self.linears(vec)
        return out
    
    def load_linears_state_dict(self, state_dict):
        self.linears.load_state_dict(state_dict)
