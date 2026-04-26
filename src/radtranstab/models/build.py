#################################################################################
# TransTabModel for Radiomics Specific Task 
#################################################################################
"""
Source: 
    https://github.com/nainye/RadiomicsRetrieval

Adapted version of the TransTab model for tabular data processing and embedding.

Original implementation:
    https://github.com/RyanWangZf/transtab

This version includes modifications for radiomics-based retrieval tasks,
including integration with anatomical positional embeddings (APE).
"""

import os
import json
from loguru import logger
import torch
import numpy as np
import pandas as pd

from radtranstab.models._embed import TransTabFeatureExtractor
from radtranstab.models._transtab import TransTabClassifier
from radtranstab.models._radtranstab import TransTabForRadiomics
import radtranstab.models.constants as constants 


def build_extractor(
    categorical_columns=None,
    numerical_columns=None,
    binary_columns=None,
    ignore_duplicate_cols=False,
    disable_tokenizer_parallel=False,
    checkpoint=None,
    **kwargs,) -> TransTabFeatureExtractor:
    '''Build a feature extractor for TransTab model.

    Parameters
    ----------
    categorical_columns: list 
        a list of categorical feature names.

    numerical_columns: list
        a list of numerical feature names.

    binary_columns: list
        a list of binary feature names, accept binary indicators like (yes,no); (true,false); (0,1).

    ignore_duplicate_cols: bool
        if there is one column assigned to more than one type, e.g., the feature age is both nominated
        as categorical and binary columns, the model will raise errors. set True to avoid this error as 
        the model will ignore this duplicate feature.

    disable_tokenizer_parallel: bool
        if the returned feature extractor is leveraged by the collate function for a dataloader,
        try to set this False in case the dataloader raises errors because the dataloader builds 
        multiple workers and the tokenizer builds multiple workers at the same time.

    checkpoint: str
        the directory of the predefined TransTabFeatureExtractor.

    Returns
    -------
    A TransTabFeatureExtractor module.

    '''
    feature_extractor = TransTabFeatureExtractor(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        binary_columns=binary_columns,
        disable_tokenizer_parallel=disable_tokenizer_parallel,
        ignore_duplicate_cols=ignore_duplicate_cols,
    )
    if checkpoint is not None:
        extractor_path = os.path.join(checkpoint, constants.EXTRACTOR_STATE_DIR)
        if os.path.exists(extractor_path):
            feature_extractor.load(extractor_path)
        else:
            feature_extractor.load(checkpoint)
    return feature_extractor

def build_classifier(
    categorical_columns=None,
    numerical_columns=None,
    binary_columns=None,
    feature_extractor=None,
    num_class=2,
    hidden_dim=128,
    num_layer=2,
    num_attention_head=8,
    hidden_dropout_prob=0,
    ffn_dim=256,
    activation='relu',
    device='cuda:0',
    checkpoint=None,
    **kwargs) -> TransTabClassifier:
    '''Build a :class:`transtab.modeling_transtab.TransTabClassifier`.

    Parameters
    ----------
    categorical_columns: list 
        a list of categorical feature names.

    numerical_columns: list
        a list of numerical feature names.

    binary_columns: list
        a list of binary feature names, accept binary indicators like (yes,no); (true,false); (0,1).
    
    feature_extractor: TransTabFeatureExtractor
        a feature extractor to tokenize the input tables. if not passed the model will build itself.

    num_class: int
        number of output classes to be predicted.

    hidden_dim: int
        the dimension of hidden embeddings.
    
    num_layer: int
        the number of transformer layers used in the encoder.
    
    num_attention_head: int
        the numebr of heads of multihead self-attention layer in the transformers.

    hidden_dropout_prob: float
        the dropout ratio in the transformer encoder.

    ffn_dim: int
        the dimension of feed-forward layer in the transformer layer.
    
    activation: str
        the name of used activation functions, support ``"relu"``, ``"gelu"``, ``"selu"``, ``"leakyrelu"``.
    
    device: str
        the device, ``"cpu"`` or ``"cuda:0"``.
    
    checkpoint: str
        the directory to load the pretrained TransTab model.

    Returns
    -------
    A TransTabClassifier model.

    '''
    model = TransTabClassifier(
        categorical_columns = categorical_columns,
        numerical_columns = numerical_columns,
        binary_columns = binary_columns,
        feature_extractor = feature_extractor,
        num_class=num_class,
        hidden_dim=hidden_dim,
        num_layer=num_layer,
        num_attention_head=num_attention_head,
        hidden_dropout_prob=hidden_dropout_prob,
        ffn_dim=ffn_dim,
        activation=activation,
        device=device,
        **kwargs,
        )
    
    if checkpoint is not None:
        model.load(checkpoint)

    return model

def build_radiomics_learner(
    categorical_columns=None,
    numerical_columns=None,
    binary_columns=None,
    feature_extractor=None,
    num_class=2,
    hidden_dim=128,
    num_layer=2,
    num_attention_head=8,
    hidden_dropout_prob=0,
    ffn_dim=256,
    projection_dim=128,
    num_sub_cols=[72, 54, 36, 18, 9, 3, 1],
    gpe_drop_rate=0.1,
    activation='relu',
    device='cuda:0',
    checkpoint=None,
    ignore_duplicate_cols=True,
    **kwargs,
    ): 
    '''Build a contrastive learning and classification model for radiomics feature extraction.

    Parameters
    ----------
    categorical_columns: list 
        a list of categorical feature names.

    numerical_columns: list
        a list of numerical feature names.

    binary_columns: list
        a list of binary feature names, accept binary indicators like (yes,no); (true,false); (0,1).
    
    feature_extractor: TransTabFeatureExtractor
        a feature extractor to tokenize the input tables. if not passed the model will build itself.

    num_class: int
        number of output classes to be predicted.

    hidden_dim: int
        the dimension of hidden embeddings.
    
    num_layer: int
        the number of transformer layers used in the encoder.
    
    num_attention_head: int
        the numebr of heads of multihead self-attention layer in the transformers.

    hidden_dropout_prob: float
        the dropout ratio in the transformer encoder.

    ffn_dim: int
        the dimension of feed-forward layer in the transformer layer.
    
    projection_dim: int
        the dimension of projection head on the top of encoder.
    
    overlap_ratio: float
        the overlap ratio of columns of different partitions when doing subsetting.
    
    num_partition: int
        the number of partitions made for vertical-partition contrastive learning.

    supervised: bool
        whether or not to take supervised VPCL, otherwise take self-supervised VPCL.
    
    temperature: float
        temperature used to compute logits for contrastive learning.

    base_temperature: float
        base temperature used to normalize the temperature.
    
    activation: str
        the name of used activation functions, support ``"relu"``, ``"gelu"``, ``"selu"``, ``"leakyrelu"``.
    
    device: str
        the device, ``"cpu"`` or ``"cuda:0"``.

    checkpoint: str
        the directory of the pretrained transtab model.
    
    ignore_duplicate_cols: bool
        if there is one column assigned to more than one type, e.g., the feature age is both nominated
        as categorical and binary columns, the model will raise errors. set True to avoid this error as 
        the model will ignore this duplicate feature.
    
    Returns
    -------
    A TransTabForRadiomics model.

    '''

    model = TransTabForRadiomics(
        categorical_columns = categorical_columns,
        numerical_columns = numerical_columns,
        binary_columns = binary_columns,
        feature_extractor=feature_extractor,
        num_class=num_class,
        hidden_dim=hidden_dim,
        num_layer=num_layer,
        num_attention_head=num_attention_head,
        hidden_dropout_prob=hidden_dropout_prob,
        ffn_dim=ffn_dim,
        projection_dim=projection_dim,
        num_sub_cols=num_sub_cols,
        gpe_drop_rate=gpe_drop_rate,
        activation=activation,
        device=device,
    )
    if checkpoint is not None:
        model.load(checkpoint)

    return model



