import json
from os import PathLike
from typing import Any, Dict, List, Union

from transformers import BertConfig, PretrainedConfig, RobertaConfig, T5Config


class PILinkModelConfig(PretrainedConfig):
    """
    Configuration class for the PILinkModel.

    Args:
        nlpl_model_config (T5Config): The configuration for the NL-PL model.
        nlnl_model_config (Union[BertConfig, RobertaConfig]): The configuration for the NL-NL model.
        linear_sizes (list, optional): The sizes of the linear layers. Defaults to [256]. We skip the last size, which is 1.
        **kwargs: Additional keyword arguments.

    Attributes:
        nlpl_model_config (T5Config): The configuration for the NL-PL model.
        nlnl_model_config (Union[BertConfig, RobertaConfig]): The configuration for the NL-NL model.
        linear_sizes (list): The sizes of the linear layers.

        Examples:

        ```python
        >>> configuration = PILinkModelConfig()

        >>> model = PILinkModel(configuration)

        >>> configuration = model.config
        ```

    """

    def __init__(self,
        nlpl_model_config: T5Config = T5Config(),
        nlnl_model_config: Union[BertConfig, RobertaConfig] = RobertaConfig(),
        linear_sizes: list = [512, 256], # last size is 1, first size is sum of NL-NL and NL-PL model hidden sizes
        **kwargs
    ):
        """
        Initializes the configuration.

        Args:
            nlpl_model_config (T5Config): The configuration for the NL-PL model.
            nlnl_model_config (Union[BertConfig, RobertaConfig]): The configuration for the NL-NL model.
            linear_sizes (list, optional): The sizes of the linear layers. Defaults to [256].
                We ignore the first size, which is sum of the NL-NL and NL-PL model hidden sizes.
                We ignore the last size, which is 1.
            **kwargs: Additional keyword arguments.
        """

        super(PILinkModelConfig, self).__init__(**kwargs)
        self.nlpl_model_config: T5Config = nlpl_model_config
        self.nlnl_model_config: Union[BertConfig, RobertaConfig] = nlnl_model_config
        self.linear_sizes: List[int] = linear_sizes

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Override the default `to_dict()` method to add the model's config.

        Returns:
            Dict[str, Any]: Dictionary of the configuration.
        """

        output = super().to_dict()
        output["nlpl_model_config"] = self.nlpl_model_config.to_dict()
        output["nlnl_model_config"] = self.nlnl_model_config.to_dict()
        output["linear_sizes"] = self.linear_sizes
        return output
    
    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string. Override the default `to_json_string()` method to add the model's config.

        Returns:
            str: String of the configuration in JSON format.
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)
    
    def to_json_file(self, json_file_path: str | PathLike, use_diff: bool = True):
        """
        Save this instance to a JSON file. Override the default `to_json_file()` method to add the model's config.

        Args:
            json_file_path (str | PathLike): Path to the JSON file.
        """
        with open(json_file_path, "w") as file:
            file.write(self.to_json_string(use_diff=use_diff))

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PILinkModelConfig":
        """
        Constructs a `PILinkModelConfig` from a Python dictionary of parameters.

        Args:
            config_dict (Dict[str, Any]): Dictionary of parameters to build the configuration from.

        Returns:
            PILinkModelConfig: Configuration object.
        """
        config_dict["nlpl_model_config"] = RobertaConfig.from_dict(config_dict["nlpl_model_config"])
        config_dict["nlnl_model_config"] = RobertaConfig.from_dict(config_dict["nlnl_model_config"])
        return cls(**config_dict)
    
    @classmethod
    def from_json_string(cls, json_string: str) -> "PILinkModelConfig":
        """
        Constructs a `PILinkModelConfig` from a JSON string of parameters.

        Args:
            json_string (str): String of parameters to build the configuration from.

        Returns:
            PILinkModelConfig: Configuration object.
        """
        return cls.from_dict(json.loads(json_string))
    
    @classmethod
    def from_json_file(cls, json_file_path: str | PathLike) -> "PILinkModelConfig":
        """
        Constructs a `PILinkModelConfig` from a JSON file of parameters.

        Args:
            json_file_path (str | PathLike): Path to the JSON file.

        Returns:
            PILinkModelConfig: Configuration object.
        """
        with open(json_file_path, "r") as file:
            return cls.from_dict(json.load(file))