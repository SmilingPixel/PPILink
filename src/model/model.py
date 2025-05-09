import pathlib
from pathlib import Path
from typing import List, Optional, Union

import torch.nn as nn
import torch
from transformers import BertModel, RobertaModel, T5EncoderModel

from .model_config import PILinkModelConfig


class PILinkModel(nn.Module):
    """
    PyTorch module for the PI Link Model.

    Attributes:
        nlpl_model (T5EncoderModel): The underlying T5 model for natural language processing (NL-PL pair).
        nlnl_model (Union[BertModel, RobertaModel]): The underlying BERT model for natural language processing (NL-NL pair).
        linears (nn.Sequential): Sequential module for linear layers.

    Methods:
        from_config: Create a new model from the specified config.
        from_trained_model: Load a trained model from a specified directory.
        from_pretrained_components: Create a new model. Initializes the NL-NL and NL-PL model from pretrained models.
        forward: Forward pass for the model.

    """

    def __init__(
        self,
        config: PILinkModelConfig,
        nlnl_model: Union[BertModel, RobertaModel],
        nlpl_model: T5EncoderModel
    ):
        super(PILinkModel, self).__init__()
        self.config: PILinkModelConfig = config

        self.nlpl_model: T5EncoderModel = nlpl_model
        self.nlnl_model: Union[BertModel, RobertaModel] = nlnl_model

        # config.linears ignore the first size, which is sum of the NL-NL and NL-PL model hidden sizes,
        # and the last size, which is 1.
        linear_block: callable = lambda in_features, out_features: nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
        )
        linear_blocks: List[nn.Sequential] = [
            linear_block(in_features, out_features)
            for in_features, out_features
            in zip([config.nlnl_model_config.hidden_size + config.nlpl_model_config.hidden_size] + config.linear_sizes, config.linear_sizes)
        ]
        linears_but_last: nn.Sequential = nn.Sequential(*linear_blocks)

        self.linears: nn.Sequential = nn.Sequential(linears_but_last, nn.Linear(([config.nlnl_model_config.hidden_size] + config.linear_sizes)[-1], 1))

    @classmethod
    def from_config(
        cls,
        config: PILinkModelConfig
    ) -> 'PILinkModel':
        """
        Create a new model from the specified config.

        Args:
            config (PILinkModelConfig): The config for the model.

        Returns:
            PILinkModel: The new model.
        """

        nlpl_model: T5EncoderModel = T5EncoderModel(config.nlpl_model_config)
        nlnl_model: Union[BertModel, RobertaModel] = RobertaModel(config.nlnl_model_config)
        return cls(config, nlnl_model, nlpl_model)
    
    @classmethod
    def from_trained_model(
        cls,
        model_name_or_path: Union[str, Path],
        device: Union[str, torch.device] = "cpu"
    ) -> 'PILinkModel':
        """
        Load a trained model from the specified directory.

        Args:
            model_name_or_path (Union[str, Path]): The path to the directory containing the model.
            device (Union[str, torch.device], optional): The device to load the model on. Defaults to "cpu".

        Returns:
            PILinkModel: The loaded trained model.

        Raises:
            ValueError: If the model directory or config file does not exist, or if the model file does not exist.
        """
        model_name_or_path = Path(model_name_or_path)
        model_dir_exist = Path.is_dir(model_name_or_path)
        if not model_dir_exist:
            raise ValueError(f"Model directory {model_name_or_path} does not exist.")

        # read config from json
        config_path = Path.joinpath(model_name_or_path, "config.json")
        config_exist = Path.is_file(config_path)
        if not config_exist:
            raise ValueError(f"Config file {config_path} does not exist.")
        config: PILinkModelConfig = PILinkModelConfig.from_json_file(config_path)

        # load model from config
        model = cls.from_config(config).to(device)

        # load model file
        model_path = Path.joinpath(model_name_or_path, "model.pt")
        model_exist = Path.is_file(model_path)
        if not model_exist:
            raise ValueError(f"Model file {model_path} does not exist.")
        model.load_state_dict(torch.load(model_path, map_location=device))

        return model.to(device)
    
    @classmethod
    def from_pretrained_components(
        cls,
        nlnl_model_name_or_path: Union[str, Path],
        nlpl_model_name_or_path: Union[str, Path],
        device: Union[str, torch.device] = "cpu"
    ) -> 'PILinkModel':
        """
        Create a new model. Initializes the NL-NL and NL-PL model from pretrained.

        Args:
            nlnl_model_name_or_path (Union[str, Path]): The name or path of the NL-NL model to load.
            nlpl_model_name_or_path (Union[str, Path]): The name or path of the NL-PL model to load.
            device (Union[str, torch.device], optional): The device to load the model on. Defaults to "cpu".
        
        Returns:
            PILinkModel: The new model.
        """
        
        nlnl_model: Union[BertModel, RobertaModel] = RobertaModel.from_pretrained(nlnl_model_name_or_path)
        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"] # ignore decoder weights if we load an encoder-decoder model
        nlpl_model: T5EncoderModel = T5EncoderModel.from_pretrained(nlpl_model_name_or_path)
        config: PILinkModelConfig = PILinkModelConfig(
            nlnl_model_config=nlnl_model.config,
            nlpl_model_config= nlpl_model.config
        )

        model = cls(
            config=config,
            nlnl_model=nlnl_model,
            nlpl_model=nlpl_model,
        ).to(device)
        return model

    def forward(self, nlnl_inputs: dict, nlpl_inputs: dict):
        """
        Forward pass for the model.

        Args:
            nlnl_inputs (Dict): The natural language pair inputs to the model, generated by tokenizer. 
                                It is a dictionary with the following keys:
                                - input_ids (torch.Tensor): The input ids of the tokens.
                                - attention_mask (torch.Tensor): The attention mask of the tokens.
                                - token_type_ids (torch.Tensor): The token type ids of the tokens. Note: RobertaModel does not need this.
            nlpl_inputs (Dict): The natural language and program language pair inputs to the model, generated by tokenizer. 
                                It is a dictionary with the following keys:
                                - input_ids (torch.Tensor): The input ids of the tokens.
                                - attention_mask (torch.Tensor): The attention mask of the tokens.
        Returns:
            torch.Tensor: The output tensor from the model.
        """
        nlnl_vec = self.nlnl_model(**nlnl_inputs).last_hidden_state
        nlnl_vec = nlnl_vec[:,0,:] # torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)
        nlpl_vec = self.nlpl_model(**nlpl_inputs).last_hidden_state
        nlpl_vec = nlpl_vec[:,0,:]
        vec = torch.cat((nlnl_vec, nlpl_vec), dim=1) # concatenate the two vectors
        out = self.linears(vec)
        return out
    