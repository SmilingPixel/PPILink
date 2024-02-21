import argparse
import datetime
import logging
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup

from dataset.pi_link_dataset import PILinkDataset
from model.model import PILinkModel
from report import log_summary, test_report
from run_util import get_arg_parser, get_logger, load_opt_sched_from_ckpt, save_model_ckpt, save_opt_sched_to_ckpt, set_seed


# use time as a unique running id
# example: 20210909123456
running_id: str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

# configure logging
log_dir: Path = Path('../logs')
log_dir.mkdir(exist_ok=True)
log_file_path: Path = log_dir.joinpath(f'{running_id}.log')
logger: logging.Logger = get_logger(__name__, log_file_path)

# configure tensorboard
tensorboard_dir: Path = Path('../tensorboard_runs')
tensorboard_dir.mkdir(exist_ok=True)
tensorboard_writer: SummaryWriter = SummaryWriter()


def train(
    dataloader: DataLoader,
    model: nn.Module,
    device: Union[torch.device, str],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler
) -> float:
    """
    Train the model.

    Args:
        dataloader (DataLoader): The data loader containing the training data.
        model (nn.Module): The model to be trained.
        device (Union[torch.device, str]): The device to be used for training.
        loss_fn (nn.Module): The loss function used for training.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler used for training.

    Returns:
        float: The average loss of the training.
    """
    start_time: float = time.time()
    model.train()
    total_size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    total_loss: float = 0.0
    for batch_idx, (nlnl_inputs, nlpl_inputs, label) in enumerate(dataloader):
        # move data to device
        for key in nlnl_inputs:
            nlnl_inputs[key] = nlnl_inputs[key].to(device)
        for key in nlpl_inputs:
            nlpl_inputs[key] = nlpl_inputs[key].to(device)
        label = label.to(device)

        optimizer.zero_grad()

        # forward
        model_output = model(nlnl_inputs, nlpl_inputs)
        # prob = torch.sigmoid(model_output)
        loss = loss_fn(model_output, label)
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
    average_loss: float = total_loss / len(dataloader)
    logger.info(f'This epoch training time: {end_time - start_time}s, average loss: {average_loss:.4f}')

    return average_loss


@torch.no_grad()
def eval(
    dataloader: DataLoader,
    model: nn.Module,
    device: Union[torch.device, str],
    loss_fn: nn.Module,
) -> Tuple[List[float], List[int], List[int], Dict[str, Any], float]:
    """
    Evaluate the model.

    Args:
        dataloader (DataLoader): The data loader for evaluation.
        model (nn.Module): The model to be evaluated.
        device (Union[torch.device, str]): The device to be used for evaluation.
        loss_fn (nn.Module): The loss function used for evaluation.

    Returns:
        pred_prob: list of predictions
        pred_labels: list of predicted labels
        true_labels: list of true labels
        eval_report: evaluation report
        average_loss: average loss
    """
    start_time: float = time.time()
    model.eval()
    total_size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    total_loss: float = 0.0
    pred_prob: List[float] = []
    true_labels: Union[List[int], List[float]] = []

    for batch_idx, (nlnl_inputs, nlpl_inputs, label) in enumerate(dataloader):
        # move data to device
        for key in nlnl_inputs:
            nlnl_inputs[key] = nlnl_inputs[key].to(device)
        for key in nlpl_inputs:
            nlpl_inputs[key] = nlpl_inputs[key].to(device)
        label = label.to(device)

        # forward
        model_output = model(nlnl_inputs, nlpl_inputs)
        prob = torch.sigmoid(model_output)
        loss = loss_fn(model_output, label)
        total_loss += loss.item()
        pred_prob.extend(prob.squeeze(1).tolist())
        true_labels.extend(label.squeeze(1).tolist())

        # output log
        if (batch_idx + 1) % 10 == 0:
            loss, current = loss.item(), min((batch_idx + 1) * batch_size, total_size)
            logger.info(f'loss: {loss:>7f} [{current:>5d}/{total_size:>5d}]')
    
    end_time: float = time.time()
    average_loss: float = total_loss / len(dataloader)
    pred_labels: List[int] = [1 if p > 0.5 else 0 for p in pred_prob]
    true_labels: List[int] = [int(l) for l in true_labels]
    eval_report: dict = test_report.generate_test_report(true_labels, pred_labels)
    accuracy: float = eval_report['accuracy']
    logger.info(f'This epoch eval time: {end_time - start_time}s, average loss: {average_loss:.4f}, accuracy: {accuracy:.4f}')
    return pred_prob, pred_labels, true_labels, eval_report, average_loss


def main():

    # set up argument parser
    parser: argparse.ArgumentParser = get_arg_parser()

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
    if args.main_model_name_or_path is not None: # load model from checkpoint
        main_model: PILinkModel = PILinkModel.from_trained_model(
            Path(args.main_model_name_or_path),
            device=device
        )
    else: # initialize a new model, and load NL-NL model and NL-PL model from pretrained model file
        main_model: PILinkModel = PILinkModel.from_pretrained_components(
            args.nlnl_model_name_or_path,
            args.nlpl_model_name_or_path,
            device=device
        )
    tensorboard_writer.add_graph(main_model)

    if not args.do_train and not args.do_test:
        raise ValueError('At least one of `do_train`, or `do_test` must be True.')
    if args.do_train and args.do_test:
        raise ValueError('Only one of `do_train`, or `do_test` can be True.')
    if args.do_eval and not args.do_train:
        raise ValueError('`do_eval` can only be True when `do_train` is True.')
    
    # initialize dataset (train or test)
    dataset_file_path: Union[str, Path] = (
        args.train_file if args.do_train
        else args.eval_file if args.do_eval
        else args.test_file
    )
    nlnl_tokenizer: Union[BertTokenizer, RobertaTokenizer] = RobertaTokenizer.from_pretrained(args.nlnl_tokenizer_name_or_path)
    nlpl_tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained(args.nlpl_tokenizer_name_or_path)
    dataset: Dataset = PILinkDataset(
        dataset_file_path,
        nlnl_model_tokenizer=nlnl_tokenizer,
        nlpl_model_tokenizer=nlpl_tokenizer,
        max_input_length=args.max_seq_length
    )
    dataloader: DataLoader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
    )

    loss_fn: nn.Module = nn.BCEWithLogitsLoss()

    # initialize eval dataset
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

        # if model is loaded from checkpoint, load optimizer and scheduler
        if args.main_model_name_or_path is not None:
            load_opt_sched_from_ckpt(args.main_model_name_or_path, optimizer, scheduler, device)

        # train
        epoch_num_max_len: int = len(str(args.num_train_epochs))
        for epoch in range(scheduler.last_epoch, args.num_train_epochs):
            logger.info(f'Epoch {epoch + 1}/{args.num_train_epochs}')
            average_train_loss: float = train(dataloader, main_model, device, loss_fn, optimizer, scheduler)

            tensorboard_writer.add_scalars(
                'Loss', 
                {'train': average_train_loss,},
                epoch + 1
            )

            # eval after each epoch of training
            if args.do_eval:
                pred_prob, pred_labels, true_labels, eval_report, average_eval_loss = eval(eval_dataloader, main_model, device, loss_fn)
                tensorboard_writer.add_scalars(
                    'Loss',
                    {'eval': average_eval_loss},
                    epoch + 1
                )

            if (epoch + 1) % args.save_steps == 0:
                save_model_ckpt(
                    ckpt_output_dir.joinpath(f'epoch_{epoch + 1:0{epoch_num_max_len}d}'),
                    main_model,
                )
                save_opt_sched_to_ckpt(
                    ckpt_output_dir.joinpath(f'epoch_{epoch + 1:0{epoch_num_max_len}d}'),
                    optimizer,
                    scheduler,
                )

        # save_final_model
        # we don't save optimizer and scheduler since training is done
        save_model_ckpt(ckpt_output_dir.joinpath(f'epoch_{args.num_train_epochs}_final'), main_model)

        # summary log
        log_summary.generate_log_summary_from_file(
            Path(log_file_path),
            ckpt_output_dir.joinpath('train_summary.png')
        )
        
    elif args.do_test:
        # output results to {output_dir}/test_results/{running_id}
        test_results_dir: Path = output_dir.joinpath('tests', str(running_id))
        test_results_dir.mkdir(parents=True, exist_ok=True)

        # torch.no_grad() is in eval(), as decorator
        pred_prob, pred_labels, true_labels, eval_report, average_eval_loss = eval(dataloader, main_model, device, loss_fn)

        # output results as json file
        test_results_file_path: Path = test_results_dir.joinpath('test_results.json')
        with open(test_results_file_path, 'w') as f:
            json.dump({
                'pred_prob': pred_prob,
                'pred_labels': pred_labels,
                'true_labels': true_labels
            }, f)

        # output report as json file
        test_report_file_path: Path = test_results_dir.joinpath('test_report.json')
        with open(test_report_file_path, 'w') as f:
            json.dump(eval_report, f, indent=2)


if __name__ == "__main__":
    # TODO: check and refactor all doc
    main()

    tensorboard_writer.close()
