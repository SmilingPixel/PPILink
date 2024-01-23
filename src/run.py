import argparse
import logging
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer

from model.model import PILinkModel
from dataset.pi_link_dataset import PILinkDataset


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def train(
    dataloader: DataLoader,
    model: nn.Module,
    device: torch.device | str,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer
) -> None:
    """
    Train the model.
    """
    model.train()
    total_size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    for batch_idx, (input1, input2, label) in enumerate(dataloader):
        for key in input1:
            input1[key] = input1[key].to(device)
        for key in input2:
            input2[key] = input2[key].to(device)
        label = label.to(device)

        # forward
        pred = model(input1, input2)
        loss = loss_fn(pred, label)

        # backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # output log
        if (batch_idx + 1) % 10 == 0:
            loss, current = loss.item(), (batch_idx + 1) * batch_size
            logger.info(f'loss: {loss:>7f} [{current:>5d}/{total_size:>5d}]')


def set_seed(seed: int) -> None:
    """
    Set all random seed for reproducibility.
    """
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():

    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument(
        "--output_dir", type=str, default='output', required=True,
        help="The output directory where the model checkpoints and test results will be saved."
    )
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
        help="The model checkpoint for weights initialization."
    )

    # dataset related
    parser.add_argument(
        "--train_file", type=str, default=None,
        help="The input training data file (json)."
    )
    parser.add_argument(
        "--eval_file", type=str, default=None,
        help="The input evaluation data file (json)."
    )
    parser.add_argument(
        "--test_file", type=str, default=None,
        help="The input test data file (json)."
    )
    parser.add_argument(
        "--max_seq_length", default=512, type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded. If left unset or set to None, this will use the predefined model maximum length"
    )

    # action config
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_test", action='store_true')

    # hyperparameters
    parser.add_argument(
        "--train_batch_size", default=4, type=int, 
        help="Batch size for training."
    )
    parser.add_argument(
        "--eval_batch_size", default=4, type=int,
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--learning_rate", default=5e-5, type=float,
        help="The initial learning rate."
    )
    parser.add_argument(
        "--num_train_epochs", default=32, type=int,
        help="Total number of training epochs."
    )

    # other
    parser.add_argument(
        '--seed',type=int, default=3407, # arXiv:2109.08203
        help='random seed'
    )

    parser.add_argument(
        '--save_steps', type=int, default=10,
        help='steps (epochs) to save model'
    )

    # running config
    parser.add_argument(
        '--device', type=str, default='cpu',
        help='device to run on, cpu, cuda or dml'
    )
    parser.add_argument(
        '--device_id', type=int, default=0,
        help='device id to run on'
    )

    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # set up output dir
    output_dir: str = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # set up device
    if args.device == 'cpu':
        device = 'cpu'
    elif args.device == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('cuda is not available')
        device = f'cuda:{args.device_id}'
    elif args.device == 'dml':
        import torch_directml
        device = torch_directml.device(args.device_id)
    else:
        raise ValueError(f'Unknown device {args.device}')
    logger.warning(f'device = {device}')

    # set up random seed
    set_seed(args.seed)

    # initialize model
    bert_model: BertModel = BertModel.from_pretrained(args.model_name_or_path).to(device)
    main_model: nn.Module = PILinkModel(bert_model, bert_model.config.hidden_size).to(device)

    # initialize dataset
    bert_tokenizer: BertTokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    dataset: Dataset = PILinkDataset(
        args.train_file,
        bert_tokenizer,
        max_input_length=args.max_seq_length
    ) # TODO: train, eval and test
    dataloader: DataLoader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
    )

    if not args.do_train and not args.do_eval and not args.do_test:
        raise ValueError('At least one of `do_train`, `do_eval` or `do_test` must be True.')
    if int(args.do_train) + int(args.do_eval) + int(args.do_test) > 1:
        raise ValueError('Only one of `do_train`, `do_eval` or `do_test` can be True.')
    
    if args.do_train:
        # freeze parameters of bert model
        for param in main_model.bert_model.parameters():
            param.requires_grad = False
        
        # set up optimizer
        optimizer: torch.optim.Optimizer = torch.optim.SGD(main_model.parameters(), lr=args.learning_rate)
        loss_fn: nn.Module = nn.BCELoss()

        # train
        for epoch in range(args.num_train_epochs):
            logger.info(f'Epoch {epoch + 1}/{args.num_train_epochs}')
            train(dataloader, main_model, device, loss_fn, optimizer)

        # save_model
        # Since we freeze parameters of BERT model, we only save parameters of linears of main_model
        # check if {output_dir}/ckpt exists
        os.makedirs(os.path.join(output_dir, 'ckpt'), exist_ok=True)
        # model name: epoch_{epoch}_lr_{lr}_bs_{bs}.pt
        torch.save(main_model.linears.state_dict(), os.path.join(output_dir, 'ckpt', f'epoch_{epoch}_lr_{args.learning_rate}_bs_{args.train_batch_size}.pt'))
    
    elif args.do_eval:
        ...
    elif args.do_test:
        ...



if __name__ == "__main__":
    main()
