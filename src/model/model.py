import torch.nn as nn
import torch
from transformers import BertModel


class PILinkModel(nn.Module):
    def __init__(self, bert_model: BertModel, bert_h_size: int):
        super(PILinkModel, self).__init__()
        self.bert_model = bert_model
        self.linears = nn.Sequential(
            nn.Linear(bert_h_size * 2, bert_h_size),
            nn.ReLU(),
            nn.Linear(bert_h_size, 1),
            nn.Sigmoid()
        )
      
    def forward(self, issue_inputs, pr_inputs):
        issue_vec = self.bert_model(**issue_inputs).last_hidden_state
        issue_vec = issue_vec[:,0,:] # torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)
        pr_vec = self.bert_model(**pr_inputs).last_hidden_state
        pr_vec = pr_vec[:,0,:] # torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)
        vec = torch.cat((issue_vec, pr_vec), dim=1)
        out = self.linears(vec)
        return out
