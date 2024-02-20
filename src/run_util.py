import argparse
import logging
import os
import random
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn


def get_arg_parser() -> argparse.ArgumentParser:
    """
    Get the argument parser for the main script.

    Returns:
        argparse.ArgumentParser: The argument parser.
    """
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument(
        "--output_dir", type=str, default='output', required=True,
        help="The output directory where the model checkpoints and test results will be saved."
    )
    parser.add_argument("--nlnl_tokenizer_name_or_path", default=None, required=True, type=str,
        help="The tokenizer used to process NL-NL pairs."
    )
    parser.add_argument("--nlpl_tokenizer_name_or_path", default=None, required=True, type=str,
        help="The tokenizer used to process NL-PL pairs."
    )
    
    # optional: load model from checkpoint
    parser.add_argument("--main_model_name_or_path", default=None, type=str,
        help="The model checkpoint dir for weights initialization. Must be provided if do_train is False. If provided, nlnl_model_name_or_path and nlpl_model_name_or_path will be ignored."
    ) # if do_train is False, this is required
    parser.add_argument("--nlnl_model_name_or_path", default=None, type=str,
        help="The model used to process NL-NL pairs into vector embeddings. It's used to initilize model if ckpt of main model is not provided."
    ) # if main_model_name_or_path is not provided, this is required
    parser.add_argument("--nlpl_model_name_or_path", default=None, type=str,
        help="The model used to process NL-PL pairs into vector embeddings. It's used to initilize model if ckpt of main model is not provided."
    ) # if main_model_name_or_path is not provided, this is required
    

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
        "--learning_rate", default=1e-5, type=float,
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
        help='Device to run on, cpu, cuda or dml. Note: bugs may exist when using dml.'
    )
    parser.add_argument(
        '--device_id', type=int, default=0,
        help='device id to run on (only works if device is "cuda")'
    )

    return parser


def save_model_ckpt(
    ckpt_output_dir: Union[str, Path],
    model: nn.Module
) -> None:
    """
    Save checkpoint, including model config.

    Args:
        ckpt_output_dir (Union[str, Path]): The directory path where the checkpoint will be saved.
        model (nn.Module): The model to be saved.
    """
    ckpt_output_dir: Path = Path(ckpt_output_dir)
    ckpt_output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_output_dir.joinpath('model.pt'))
    model.config.to_json_file(ckpt_output_dir.joinpath('config.json'))


def save_opt_sched_to_ckpt(
    ckpt_output_dir: Union[str, Path],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    """
    Save optimizer and scheduler to checkpoint.

    Args:
        ckpt_output_dir (Union[str, Path]): The directory path where the checkpoint will be saved.
        optimizer (torch.optim.Optimizer): The optimizer to be saved.
        scheduler (torch.optim.lr_scheduler.LRScheduler): The scheduler to be saved.
    """
    ckpt_output_dir: Path = Path(ckpt_output_dir)
    ckpt_output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(optimizer.state_dict(), ckpt_output_dir.joinpath('optimizer.pt'))
    torch.save(scheduler.state_dict(), ckpt_output_dir.joinpath('scheduler.pt'))


def load_opt_sched_from_ckpt(
    ckpt_output_dir: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    device: Union[torch.device, str] = 'cpu'
) -> None:
    """
    Load checkpoint.

    Args:
        ckpt_output_dir (Union[str, Path]): The directory path where the checkpoint is saved.
        optimizer (Optional[torch.optim.Optimizer], optional): The optimizer to be loaded. Defaults to None.
        scheduler (Optional[torch.optim.lr_scheduler.LRScheduler], optional): The scheduler to be loaded. Defaults to None.
        device (Union[torch.device, str], optional): The device to be used for loading. Defaults to 'cpu'.
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


def get_logger(logger_name: str, log_file_path: Optional[Path] = None) -> logging.Logger:
    """
    Get a logger.
    The returned logger output to console only. If log_file is provided, it will also output to the file.

    Args:
        logger_name (str): The name of the logger.
        log_file_path (Optional[Path], optional): The log file. Defaults to None.

    Returns:
        logging.Logger: The logger.
    """
    
    logging_format: logging.Formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S'
    )
    logger: logging.Logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    file_handler: logging.FileHandler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging_format)
    consola_handler: logging.StreamHandler = logging.StreamHandler()
    consola_handler.setFormatter(logging_format)
    logger.addHandler(file_handler)
    logger.addHandler(consola_handler)

    return logger
