import argparse
import datetime
import logging
import os
import json
import random
import time
from pathlib import Path
from typing import Any, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer, T5Tokenizer, get_linear_schedule_with_warmup

from dataset.pi_link_dataset import PILinkDataset
from model.model import PILinkModel
from report import log_summary, test_report


# use time as a unique running id
# example: 20210909123456
running_id: str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

# configure logging
Path('logs').mkdir(exist_ok=True)
logging_format: logging.Formatter = logging.Formatter(
    fmt='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_file_path_str: str = f'logs/{running_id}.log'
file_handler: logging.FileHandler = logging.FileHandler(log_file_path_str)
file_handler.setFormatter(logging_format)
consola_handler: logging.StreamHandler = logging.StreamHandler()
consola_handler.setFormatter(logging_format)
logger.addHandler(file_handler)
logger.addHandler(consola_handler)


def train(
    dataloader: DataLoader,
    model: nn.Module,
    device: Union[torch.device, str],
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
    for batch_idx, (nlnl_inputs, nlpl_inputs, label) in enumerate(dataloader):
        for key in nlnl_inputs:
            nlnl_inputs[key] = nlnl_inputs[key].to(device)
        for key in nlpl_inputs:
            nlpl_inputs[key] = nlpl_inputs[key].to(device)
        label = label.to(device)

        optimizer.zero_grad()

        # forward
        pred = model(nlnl_inputs, nlpl_inputs)
        loss = loss_fn(pred, label)
        total_loss += loss.item() * label.size(0)

        # backward
        loss.backward()
        optimizer.step()

        # output log
        if (batch_idx + 1) % 10 == 0:
            loss, current = loss.item(), min((batch_idx + 1) * batch_size, total_size)
            logger.info(f'loss: {loss:>7f} [{current:>5d}/{total_size:>5d}]')
    
    scheduler.step()
    end_time: float = time.time()
    average_loss: float = total_loss / len(dataloader)
    logger.info(f'This epoch training time: {end_time - start_time}s, average loss: {average_loss:.4f}')


def eval(
    dataloader: DataLoader,
    model: nn.Module,
    device: Union[torch.device, str],
    loss_fn: nn.Module,
) -> None:
    """
    Evaluate the model.
    """
    start_time: float = time.time()
    model.eval()
    total_size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    total_loss: float = 0.0
    with torch.no_grad():
        for batch_idx, (nlnl_inputs, nlpl_inputs, label) in enumerate(dataloader):
            for key in nlnl_inputs:
                nlnl_inputs[key] = nlnl_inputs[key].to(device)
            for key in nlpl_inputs:
                nlpl_inputs[key] = nlpl_inputs[key].to(device)
            label = label.to(device)

            # forward
            pred = model(nlnl_inputs, nlpl_inputs)
            loss = loss_fn(pred, label)
            total_loss += loss.item() * label.size(0)

            # output log
            if (batch_idx + 1) % 10 == 0:
                loss, current = loss.item(), min((batch_idx + 1) * batch_size, total_size)
                logger.info(f'loss: {loss:>7f} [{current:>5d}/{total_size:>5d}]')
    
    end_time: float = time.time()
    average_loss: float = total_loss / len(dataloader)
    logger.info(f'This epoch eval time: {end_time - start_time}s, average loss: {average_loss:.4f}')


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
        for batch_idx, (nlnl_inputs, nlpl_inputs, label) in enumerate(dataloader):
            for key in nlnl_inputs:
                nlnl_inputs[key] = nlnl_inputs[key].to(device)
            for key in nlpl_inputs:
                nlpl_inputs[key] = nlpl_inputs[key].to(device)
            label = label.to(device)

            # forward
            pred = model(nlnl_inputs, nlpl_inputs)
            all_pred.extend(pred.squeeze(1).tolist())
            true_labels.extend(label.squeeze(1).tolist())

            # output log
            if (batch_idx + 1) % 10 == 0:
                current = min((batch_idx + 1) * batch_size, total_size)
                logger.info(f'[{current:>5d}/{total_size:>5d}]')
    
    end_time: float = time.time()
    logger.info(f'This epoch testing time: {end_time - start_time}s')
    return all_pred, true_labels


def save_ckpt(
    ckpt_output_dir: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> None:
    """
    Save checkpoint.
    """
    ckpt_output_dir: Path = Path(ckpt_output_dir)
    ckpt_output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_output_dir.joinpath('model.pt'))
    model.config.to_json_file(ckpt_output_dir.joinpath('config.json'))
    if optimizer is not None:
        torch.save(optimizer.state_dict(), ckpt_output_dir.joinpath('optimizer.pt'))
    if scheduler is not None:
        torch.save(scheduler.state_dict(), ckpt_output_dir.joinpath('scheduler.pt'))


def load_opt_sched_from_ckpt(
    ckpt_output_dir: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    device: Union[torch.device, str] = 'cpu'
) -> None:
    """
    Load checkpoint.
    """
    ckpt_output_dir: Path = Path(ckpt_output_dir)
    if optimizer is not None:
        optimizer.load_state_dict(torch.load(ckpt_output_dir.joinpath('optimizer.pt'), map_location=device))
    if scheduler is not None:
        scheduler.load_state_dict(torch.load(ckpt_output_dir.joinpath('scheduler.pt'), map_location=device))
        

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
    parser.add_argument("--nlnl_model_name_or_path", default=None, required=True, type=str,
        help="The model used to process NL-NL pairs into vector embeddings. It's used to initilize model if ckpt of main model is not provided."
    ) # required=True because it's used to provide tokenizer fo NL-NL
    parser.add_argument("--nlpl_model_name_or_path", default=None, required=True, type=str,
        help="The model used to process NL-PL pairs into vector embeddings. It's used to initilize model if ckpt of main model is not provided."
    ) # required=True because it's used to provide tokenizer for NL-PL

    # optional: load model from checkpoint
    parser.add_argument("--main_model_name_or_path", default=None, type=str,
        help="The model checkpoint dir for weights initialization. If do_train, ckpt of optimizer and scheduler should also be provided."
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
        help="The maximum total input sequence length after tokenization (max(len(nlnl_pair_tokens, nlpl_pair_tokens))). Sequences longer than this will be truncated, sequences shorter will be padded. If left unset or set to None, this will use the predefined model maximum length"
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
        "--learning_rate", default=3e-5, type=float,
        help="The peak learning rate."
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
        "--num_train_epochs", default=24, type=int,
        help="Total number of training epochs."
    )

    # other
    parser.add_argument(
        '--seed',type=int, default=3407, # arXiv:2109.08203
        help='random seed'
    )

    parser.add_argument(
        '--save_steps', type=int, default=6,
        help='steps (epochs) to save model'
    )

    # running config
    parser.add_argument(
        '--device', type=str, default='cpu',
        help='device to run on, cpu, cuda or dml'
    )
    parser.add_argument(
        '--device_id', type=int, default=0,
        help='device id to run on (only works if device is "cuda")'
    )

    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # set up output dir
    output_dir: Path = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

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

    # check params and load model
    if args.main_model_name_or_path is None and (args.nlnl_model_name_or_path is None or args.nlpl_model_name_or_path is None):
        raise ValueError('If main_model_name_or_path is not provided, both nlnl_model_name_or_path and nlpl_model_name_or_path must be provided.')
    # initialize model
    # TODO: pr code model
    if args.main_model_name_or_path is not None: # load model from checkpoint
        main_model: PILinkModel = PILinkModel.from_trained_model(
            Path(args.main_model_name_or_path),
            device=device
        )
    else: # initialize from scratch, and load NL-NL model and NL-PL model from pretrained model file
        main_model: PILinkModel = PILinkModel.from_scratch(
            args.nlnl_model_name_or_path,
            args.nlpl_model_name_or_path,
            device=device
        )

    if not args.do_train and not args.do_test:
        raise ValueError('At least one of `do_train`, or `do_test` must be True.')
    if int(args.do_train) + int(args.do_test) > 1:
        raise ValueError('Only one of `do_train`, or `do_test` can be True.')
    if args.do_eval and not args.do_train:
        raise ValueError('`do_eval` can only be True when `do_train` is True.')
    
    # initialize dataset
    file_path: Union[str, Path] = (
        args.train_file if args.do_train
        else args.eval_file if args.do_eval
        else args.test_file
    )
    nlnl_tokenizer: Union[BertTokenizer, RobertaTokenizer] = RobertaTokenizer.from_pretrained(args.nlnl_model_name_or_path)
    nlpl_tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained(args.nlpl_model_name_or_path)
    dataset: Dataset = PILinkDataset(
        file_path,
        nlnl_model_tokenizer=nlnl_tokenizer,
        nlpl_model_tokenizer=nlpl_tokenizer,
        max_input_length=args.max_seq_length
    )
    dataloader: DataLoader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
    )

    if args.do_eval:
        eval_dataset: Dataset = PILinkDataset(
            args.eval_file,
            nlnl_model_tokenizer=nlnl_tokenizer,
            nlpl_model_tokenizer=nlpl_tokenizer,
            max_input_length=args.max_seq_length
        )
        eval_dataloader: DataLoader = DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
        )

    if args.do_train:

        # check if {output_dir}/ckpt/{running_id} exists
        ckpt_output_dir: Path = output_dir.joinpath('ckpt', str(running_id))
        ckpt_output_dir.mkdir(parents=True, exist_ok=True)
        
        # set up optimizer, scheduler and loss function
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            params=main_model.parameters(),
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
        loss_fn: nn.Module = nn.BCELoss() # TODO: replace it with BCEwithLogitsLoss

        # if model is loaded from checkpoint, load optimizer and scheduler
        if args.main_model_name_or_path is not None:
            load_opt_sched_from_ckpt(args.main_model_name_or_path, optimizer, scheduler, device)

        # train
        epoch_num_max_len: int = len(str(args.num_train_epochs))
        for epoch in range(scheduler.last_epoch, args.num_train_epochs):
            logger.info(f'Epoch {epoch + 1}/{args.num_train_epochs}')
            train(dataloader, main_model, device, loss_fn, optimizer, scheduler)

            if args.do_eval:
                eval(eval_dataloader, main_model, device, loss_fn)

            if (epoch + 1) % args.save_steps == 0:
                save_ckpt(
                    ckpt_output_dir.joinpath(f'epoch_{epoch + 1:0{epoch_num_max_len}d}'),
                    main_model,
                    optimizer,
                    scheduler
                )

        # save_final_model
        # we don't save optimizer and scheduler since training is done
        save_ckpt(ckpt_output_dir.joinpath(f'epoch_{args.num_train_epochs}_final'), main_model)

        # summary log
        log_summary.generate_log_summary_from_file(
            Path(log_file_path_str),
            ckpt_output_dir.joinpath('train_summary.png')
        )
        
    elif args.do_test:
        # output results to {output_dir}/test_results/{running_id}
        test_results_dir: Path = output_dir.joinpath('tests', str(running_id))
        test_results_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            preds, true_labels = test(dataloader, main_model, device)

            pred_prob: List[float] = preds
            pred_labels: List[int] = [1 if p > 0.5 else 0 for p in pred_prob]

            # output as json file
            test_results_file_path: Path = test_results_dir.joinpath('test_results.json')
            with open(test_results_file_path, 'w') as f:
                json.dump({
                    'pred_prob': pred_prob,
                    'pred_labels': pred_labels,
                    'true_labels': true_labels
                }, f)

            # generate test report
            test_report.generate_test_report(
                test_results_file_path,
                test_results_dir.joinpath('test_report.json')
            )


if __name__ == "__main__":
    # TODO: check and refactor all doc
    main()
