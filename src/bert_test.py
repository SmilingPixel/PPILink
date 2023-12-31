import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
import torch_directml
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer
from transformers import AutoModel, BertModel

from datasets.test_dataset import TestDataset


class BertClassifier(nn.Module):
    def __init__(self, bert_model: BertModel):
        super(BertClassifier, self).__init__()
        self.bert = bert_model
        # BERT-small hidden state size is 512
        self.linear = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        # initialing weights and bias
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, **inputs):
        # get last_hidden_state
        vec = self.bert(**inputs).last_hidden_state
        # print("vec shape: ", vec.shape)
        # only get first token 'cls'
        vec = vec[:,0,:]
        vec = vec.view(-1, 512)

        out = self.linear(vec)
        out = self.sigmoid(out)
        return out


def main():
    device = torch_directml.device(0)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    raw_dataset = [
        ("zero dog hi world", 0),
        ("zero cat hello china", 1),
        ("one dog good morning", 0),
        ("one cat love you", 1),
    ]

    tokenizer = BertTokenizer(vocab_file='./models/bert-small/vocab.txt')
    model = BertModel.from_pretrained('./models/bert-small')
    
    test_dataset = TestDataset(raw_dataset, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    classifier = BertClassifier(model)

    # First, turn off the gradient for all parameters.
    for param in classifier.parameters():
        param.requires_grad = False

    # Second, turn on only last BERT layer.
    for param in classifier.bert.encoder.layer[-1].parameters():
        param.requires_grad = True

    # Finally, turn on classifier layer.
    for param in classifier.linear.parameters():
        param.requires_grad = True

    import torch.optim as optim

    # The pre-learned sections should have a smaller learning rate, and the last total combined layer should be larger.
    optimizer = optim.Adam([
        {'params': classifier.bert.encoder.layer[-1].parameters(), 'lr': 0.0000001},
        {'params': classifier.linear.parameters(), 'lr': 0.0000001}
    ])

    # loss function, binary cross-entropy
    loss_function = nn.BCELoss()

    # send network to GPU
    classifier.to(device, dtype=torch.float32)
    losses = []

    for epoch in range(40):
        all_loss = 0.0
        classifier.train()
        for batch, (inputs_map, y) in enumerate(test_dataloader):
            # classifier.zero_grad()
            y = y.to(device, dtype=torch.float32)
            for key in inputs_map:
                inputs_map[key] = inputs_map[key].to(device)
            out = classifier(**inputs_map)
            # print(f'out: {out}, y: {y}')
            loss = loss_function(out, y)
            loss.backward()
            # print(f'grad: {classifier.linear.weight.grad[0, :5]}, lenear bias: {classifier.linear.bias}')
            optimizer.step()
            optimizer.zero_grad()
            all_loss += loss.item()
        print("epoch", epoch, "\t" , "loss", all_loss)

    
    answer = []
    prediction = []
    with torch.no_grad():
        for (inputs_map, y) in test_dataloader:

            for key in inputs_map:
                inputs_map[key] = inputs_map[key].to(device)
            label_tensor = y.to(device)

            score = classifier(**inputs_map).squeeze(0)
            print("score: ", score)
            pred = int(score.cpu().numpy()[0] > 0.5)

            prediction.append(pred)
            answer += list(label_tensor.cpu().numpy()[0])
    print("prediction: ", prediction)
    print("answer: ", answer)



if __name__ == "__main__":
    main()