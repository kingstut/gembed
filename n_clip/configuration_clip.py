""" CLIP model configuration"""

class CLIPTextConfig():
    r"""
    copy of CLIPTextConfig
    """

    def __init__(
        self,
        vocab_size=49408,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=8,
        max_position_embeddings=77,
        hidden_act="quick_gelu",
        layer_norm_eps=0.00001,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout


class CLIPVisionConfig():
    r"""
    copy of  CLIPVisionConfig
    ```"""

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=32,
        hidden_act="quick_gelu",
        layer_norm_eps=0.00001,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act


class n_CLIPConfig():
    r"""
    [`n_CLIPConfig`] is the configuration class to store the configuration of a [`n_CLIPModel`]. It is used to instantiate
    CLIP model according to the specified arguments, defining the n model configs.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        list_config_dicts (`dict`, *optional*):
            List of dictionaries of configuration options used to initialize [`n_CLIPVisionConfig`] or [`n_CLIPTextConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* paramter. Default is used as per the original CLIP implementation.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import n_CLIPConfig, n_CLIPModel

    >>> # Initializing a n_CLIPConfig 
    >>> configuration = n_CLIPConfig()

    >>> # Initializing a n_CLIPModel (with random weights)
    >>> model = CLIPModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "n_clip"
    is_composition = True

    def __init__(
        self,
        list_config_dicts=None,
        projection_dim=512,
        logit_scale_init_value=2.6592,
    ):
        super().__init__()

        self.configs = [CLIPTextConfig(**config_dict) if config_dict["type"]=="text" 
                else CLIPVisionConfig(**config_dict)
                for config_dict in list_config_dicts]

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0

