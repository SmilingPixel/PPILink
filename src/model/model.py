import torch.nn as nn
import torch
from transformers import BertModel

BERT_SMALL_H = 512
BERT_BASE_H = 768

class Model(nn.Module):
    def __init__(self, bert_model: BertModel):
        super(Model, self).__init__()
        self.bert_model = bert_model
        self.linears = nn.Sequential(
            nn.Linear(BERT_SMALL_H * 2, BERT_SMALL_H),
            nn.ReLU(),
            nn.Linear(BERT_SMALL_H, 1),
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
