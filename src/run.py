import argparse
import datetime
import logging
import os
import pathlib
import random
import time
from typing import Any, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer, RobertaModel, get_linear_schedule_with_warmup

from model.model import PILinkModel
from dataset.pi_link_dataset import PILinkDataset


# use time as a unique running id
# example: 20210909123456
running_id: str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

# configure logging
os.makedirs('logs', exist_ok=True)
logging_format: logging.Formatter = logging.Formatter(
    fmt='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler: logging.FileHandler = logging.FileHandler(f'logs/{running_id}.log')
file_handler.setFormatter(logging_format)
consola_handler: logging.StreamHandler = logging.StreamHandler()
consola_handler.setFormatter(logging_format)
logger.addHandler(file_handler)
logger.addHandler(consola_handler)


def train(
    dataloader: DataLoader,
    model: nn.Module,
    device: torch.device | str,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler
) -> None:
    """
    Train the model.
    """
    start_time: float = time.time()
    model.train()
    total_size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    total_loss: float = 0.0
    for batch_idx, (inputs, label) in enumerate(dataloader):
        for key in inputs:
            inputs[key] = inputs[key].to(device)
        label = label.to(device)

        optimizer.zero_grad()

        # forward
        pred = model(inputs)
        loss = loss_fn(pred, label)
        total_loss += loss.item()

        # backward
        loss.backward()
        optimizer.step()

        # output log
        if (batch_idx + 1) % 10 == 0:
            loss, current = loss.item(), min((batch_idx + 1) * batch_size, total_size)
            logger.info(f'loss: {loss:>7f} [{current:>5d}/{total_size:>5d}]')
    
    scheduler.step()
    end_time: float = time.time()
    logger.info(f'This epoch training time: {end_time - start_time}s, total loss: {total_loss:.4f}')


def test(
    dataloader: DataLoader,
    model: nn.Module,
    device: torch.device | str,
) -> Tuple[List[float], List[float]]:
    """
    Test the model.

    Returns:
        all_pred: list of predictions
        true_labels: list of true labels
    """
    start_time: float = time.time()
    model.eval()
    total_size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    all_pred: list[float] = []
    true_labels: list[float] = []
    with torch.no_grad():
        for batch_idx, (inputs, label) in enumerate(dataloader):
            for key in inputs:
                inputs[key] = inputs[key].to(device)
            label = label.to(device)

            # forward
            pred = model(inputs)
            all_pred.append(pred.item())
            true_labels.append(label.item())

            # output log
            if (batch_idx + 1) % 10 == 0:
                current = min((batch_idx + 1) * batch_size, total_size)
                logger.info(f'[{current:>5d}/{total_size:>5d}]')
    
    end_time: float = time.time()
    logger.info(f'This epoch testing time: {end_time - start_time}s')
    return all_pred, true_labels


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
    parser.add_argument("--nlp_model_name_or_path", default=None, type=str, required=True,
        help="The model used to process nl into vector embeddings."
    )
    parser.add_argument("--code_model_name_or_path", default=None, type=str, required=True,
        help="The model used to process code into vector embeddings."
    )

    # optional: load model from checkpoint
    parser.add_argument("--main_model_name_or_path", default=None, type=str,
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
        "--train_batch_size", default=32, type=int, 
        help="Batch size for training."
    )
    parser.add_argument(
        "--eval_batch_size", default=32, type=int,
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--learning_rate", default=0.002, type=float,
        help="The initial learning rate."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float,
        help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--adam_weight_decay", default=0.01, type=float,
        help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_beta1", default=0.9, type=float,
        help="Beta1 for Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2", default=0.999, type=float,
        help="Beta2 for Adam optimizer."
    )
    parser.add_argument(
        "--warmup_steps", default=4, type=int,
        help="Linear warmup over warmup_steps."
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
    # TODO: pr code model
    # code_model: RobertaModel = RobertaModel.from_pretrained(args.code_model_name_or_path).to(device)
    code_model: Any = None
    nlp_model: BertModel = BertModel.from_pretrained(args.nlp_model_name_or_path).to(device)
    main_model: nn.Module = PILinkModel(
        code_model,
        nlp_model,
    ).to(device)

    if not args.do_train and not args.do_eval and not args.do_test:
        raise ValueError('At least one of `do_train`, `do_eval` or `do_test` must be True.')
    if int(args.do_train) + int(args.do_eval) + int(args.do_test) > 1:
        raise ValueError('Only one of `do_train`, `do_eval` or `do_test` can be True.')
    
    # initialize dataset
    file_path: Union[str, pathlib.Path] = (
        args.train_file if args.do_train
        else args.eval_file if args.do_eval
        else args.test_file
    )
    bert_tokenizer: BertTokenizer = BertTokenizer.from_pretrained(args.nlp_model_name_or_path)
    dataset: Dataset = PILinkDataset(
        file_path,
        bert_tokenizer,
        max_input_length=args.max_seq_length
    ) # TODO: train, eval and test
    dataloader: DataLoader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
    )
    
    if args.do_train:
        # freeze all parameters but those of linears
        for param in main_model.parameters():
            param.requires_grad = False
        for param in main_model.linears.parameters():
            param.requires_grad = True
        
        # set up optimizer, scheduler and loss function
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            [
                {'params': main_model.linears.parameters()},
            ],
            lr=args.learning_rate,
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
        )
        scheduler: torch.optim.lr_scheduler.LRScheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.num_train_epochs,
        )
        loss_fn: nn.Module = nn.BCELoss()

        # train
        for epoch in range(args.num_train_epochs):
            logger.info(f'Epoch {epoch + 1}/{args.num_train_epochs}')
            train(dataloader, main_model, device, loss_fn, optimizer, scheduler)

        # save_model
        # Since we freeze parameters of BERT model, we only save parameters of linears of main_model
        # check if {output_dir}/ckpt exists
        os.makedirs(os.path.join(output_dir, 'ckpt'), exist_ok=True)
        # model name: epoch_{epoch}_lr_{lr}_bs_{bs}.pt
        torch.save(main_model.linears.state_dict(), os.path.join(output_dir, 'ckpt', f'epoch_{args.num_train_epochs}_lr_{args.learning_rate}_bs_{args.train_batch_size}.pt'))
    
    elif args.do_eval:
        ...
    elif args.do_test:
        with torch.no_grad():
            preds, true_labels = test(dataloader, main_model, device)
            # TODO: calculate metrics
            pass

        



if __name__ == "__main__":
    main()
