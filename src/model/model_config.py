import json
from os import PathLike
from typing import Any, Dict, Union

from transformers import BertConfig, PretrainedConfig, RobertaConfig


class PILinkModelConfig(PretrainedConfig):
    """
    Configuration class for the PILinkModel.

    Args:
        code_model_config (RobertaConfig): The configuration for the code model. TODO: Add more details.
        nlp_model_config (BertConfig): The configuration for the NLP model.
        num_linear_layers (int, optional): The number of linear layers. Defaults to 2.
        linear_sizes (list, optional): The sizes of the linear layers. Defaults to [256]. We skip the last size, which is 1.
        **kwargs: Additional keyword arguments.

    Attributes:
        code_model_config (RobertaConfig): The configuration for the code model.
        nlp_model_config (BertConfig): The configuration for the NLP model.
        num_linear_layers (int): The number of linear layers.
        linear_sizes (list): The sizes of the linear layers.

        Examples:

        ```python
        >>> configuration = PILinkModelConfig()

        >>> model = PILinkModel(configuration)

        >>> configuration = model.config
        ```

    """

    def __init__(self,
        code_model_config: RobertaConfig = RobertaConfig(), # TODO
        nlp_model_config: BertConfig = BertConfig(),
        num_linear_layers: int = 2,
        linear_sizes: list = [256], # last size is 1
        **kwargs
    ):
        """
        Initializes the configuration.

        Args:
            code_model_config (RobertaConfig): The configuration for the code model. TODO: Add more details.
            nlp_model_config (BertConfig): The configuration for the NLP model.
            num_linear_layers (int, optional): The number of linear layers. Defaults to 2.
            linear_sizes (list, optional): The sizes of the linear layers. Defaults to [256].
                We ignore the first size, which is sum of the code and NLP model hidden sizes.
                We ignore the last size, which is 1.
            **kwargs: Additional keyword arguments.
        """

        super(PILinkModelConfig, self).__init__(**kwargs)
        self.code_model_config = code_model_config
        self.nlp_model_config = nlp_model_config
        assert num_linear_layers > 0, "Number of linear layers must be greater than 0."
        assert num_linear_layers == len(linear_sizes) + 1, "Number of linear layers must match the number of sizes."
        self.num_linear_layers = num_linear_layers
        self.linear_sizes = linear_sizes

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Override the default `to_dict()` method to add the model's config.

        Returns:
            Dict[str, Any]: Dictionary of the configuration.
        """

        output = super().to_dict()
        output["code_model_config"] = self.code_model_config.to_dict()
        output["nlp_model_config"] = self.nlp_model_config.to_dict()
        output["num_linear_layers"] = self.num_linear_layers
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
        config_dict["code_model_config"] = RobertaConfig.from_dict(config_dict["code_model_config"])
        config_dict["nlp_model_config"] = BertConfig.from_dict(config_dict["nlp_model_config"])
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