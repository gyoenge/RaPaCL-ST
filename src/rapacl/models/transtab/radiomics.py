import os
import json
import math
import collections
from typing import List, Dict, Optional

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.init as nn_init

from src.rapacl.models.transtab.base import (
    TransTabFeatureExtractor,
    TransTabFeatureProcessor,
    TransTabInputEncoder,
    TransTabEncoder,
)
import src.rapacl.utils.constants as constants

import logging
logger = logging.getLogger(__name__)


class AdditionalToken(nn.Module):
    """Add a learnable token at the first position of the sequence.

    Call order matters:
        contrastive_token first, then cls_token
        => final token order becomes [CLS, CONTRAST, feature...]
    """
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(Tensor(hidden_dim))
        nn_init.uniform_(self.weight, a=-1 / math.sqrt(hidden_dim), b=1 / math.sqrt(hidden_dim))
        self.hidden_dim = hidden_dim

    def expand(self, *leading_dimensions):
        new_dims = (1,) * (len(leading_dimensions) - 1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

    def forward(self, embedding, attention_mask=None, **kwargs):
        embedding = torch.cat([self.expand(len(embedding), 1), embedding], dim=1)
        outputs = {"embedding": embedding}
        if attention_mask is not None:
            attention_mask = torch.cat(
                [torch.ones(attention_mask.shape[0], 1, device=attention_mask.device), attention_mask],
                dim=1,
            )
        outputs["attention_mask"] = attention_mask
        return outputs


class TransTabModelCustom(nn.Module):
    """TransTab encoder with optional CLS / contrastive token separation."""
    def __init__(
        self,
        separate_contrast_token: bool = True,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        feature_extractor=None,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0.1,
        ffn_dim=256,
        activation="relu",
        device="cuda:0",
        **kwargs,
    ) -> None:
        super().__init__()

        self.categorical_columns = list(set(categorical_columns)) if categorical_columns is not None else None
        self.numerical_columns = list(set(numerical_columns)) if numerical_columns is not None else None
        self.binary_columns = list(set(binary_columns)) if binary_columns is not None else None

        if feature_extractor is None:
            feature_extractor = TransTabFeatureExtractor(
                categorical_columns=self.categorical_columns,
                numerical_columns=self.numerical_columns,
                binary_columns=self.binary_columns,
                **kwargs,
            )

        feature_processor = TransTabFeatureProcessor(
            vocab_size=feature_extractor.vocab_size,
            pad_token_id=feature_extractor.pad_token_id,
            hidden_dim=hidden_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            device=device,
        )

        self.input_encoder = TransTabInputEncoder(
            feature_extractor=feature_extractor,
            feature_processor=feature_processor,
            device=device,
        )

        self.encoder = TransTabEncoder(
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
        )

        self.cls_token = AdditionalToken(hidden_dim=hidden_dim)
        self.contrastive_token = AdditionalToken(hidden_dim=hidden_dim) if separate_contrast_token else None

        self.hidden_dim = hidden_dim
        self.separate_contrast_token = separate_contrast_token
        self.device = device
        self.to(device)

    def forward(self, x):
        embedded = self.input_encoder(x)

        if self.contrastive_token is not None:
            embedded = self.contrastive_token(**embedded)
        embedded = self.cls_token(**embedded)

        encoder_output = self.encoder(**embedded)

        cls_embedding = encoder_output[:, 0, :]
        contrastive_embedding = encoder_output[:, 1, :] if self.contrastive_token is not None else None

        return {
            "cls_embedding": cls_embedding,
            "contrastive_embedding": contrastive_embedding,
            "encoder_output": encoder_output,
        }

    def load(self, ckpt_dir):
        model_name = os.path.join(ckpt_dir, constants.WEIGHTS_NAME)
        state_dict = torch.load(model_name, map_location="cpu")
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        logger.info(f"[load model] missing keys: {missing_keys}")
        logger.info(f"[load model] unexpected keys: {unexpected_keys}")
        logger.info(f"[load model] load model from: {ckpt_dir}")

        self.input_encoder.feature_extractor.load(os.path.join(ckpt_dir, constants.EXTRACTOR_STATE_DIR))
        self.binary_columns = self.input_encoder.feature_extractor.binary_columns
        self.categorical_columns = self.input_encoder.feature_extractor.categorical_columns
        self.numerical_columns = self.input_encoder.feature_extractor.numerical_columns

    def save(self, ckpt_dir):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)

        torch.save(self.state_dict(), os.path.join(ckpt_dir, constants.WEIGHTS_NAME))

        if self.input_encoder.feature_extractor is not None:
            self.input_encoder.feature_extractor.save(ckpt_dir)

        state_dict_input_encoder = self.input_encoder.state_dict()
        torch.save(state_dict_input_encoder, os.path.join(ckpt_dir, constants.INPUT_ENCODER_NAME))


class TransTabRadiomicsEncoder(TransTabModelCustom):
    """Radiomics-specific TransTab encoder.

    Responsibilities:
    - build random multi-view feature subsets
    - encode full-view / sub-view radiomics tables
    - return CLS embedding and multi-view contrastive embeddings

    Does NOT:
    - compute losses
    - own classifier heads
    - own batch correction discriminator
    """
    def __init__(
        self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        feature_extractor=None,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0.1,
        ffn_dim=256,
        num_sub_cols: List[int] = [93, 62, 31, 15, 7, 3, 1],
        separate_contrast_token: bool = True,
        activation="relu",
        device="cuda:0",
        **kwargs,
    ) -> None:
        super().__init__(
            separate_contrast_token=separate_contrast_token,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            feature_extractor=feature_extractor,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            device=device,
            **kwargs,
        )

        self.num_sub_cols = num_sub_cols
        self.activation = activation
        self.device = device
        self.to(device)

    def forward(self, x: pd.DataFrame):
        if not isinstance(x, pd.DataFrame):
            raise ValueError(f"expect input x to be pd.DataFrame, get {type(x)} instead")

        sub_x_list = self._build_sub_x_list_random(x, self.num_sub_cols)

        multiview_embeddings = []
        cls_embedding_full = None
        contrastive_embedding_full = None

        for i, sub_x in enumerate(sub_x_list):
            out = super().forward(sub_x)

            if i == 0:
                cls_embedding_full = out["cls_embedding"]
                contrastive_embedding_full = out["contrastive_embedding"]

            if self.contrastive_token is not None:
                view_embedding = out["contrastive_embedding"]
            else:
                view_embedding = out["cls_embedding"]

            multiview_embeddings.append(view_embedding)

        multiview_embeddings = torch.stack(multiview_embeddings, dim=1)  # [B, V, H]

        return {
            "cls_embedding": cls_embedding_full,                    # [B, H]
            "contrastive_embedding": contrastive_embedding_full,    # [B, H] or None
            "multiview_embeddings": multiview_embeddings,           # [B, V, H]
        }

    def forward_with_subx(self, sub_x_list: List[pd.DataFrame]):
        multiview_embeddings = []
        cls_embedding_full = None
        contrastive_embedding_full = None

        for i, sub_x in enumerate(sub_x_list):
            out = super().forward(sub_x)

            if i == 0:
                cls_embedding_full = out["cls_embedding"]
                contrastive_embedding_full = out["contrastive_embedding"]

            if self.contrastive_token is not None:
                view_embedding = out["contrastive_embedding"]
            else:
                view_embedding = out["cls_embedding"]

            multiview_embeddings.append(view_embedding)

        multiview_embeddings = torch.stack(multiview_embeddings, dim=1)

        return {
            "cls_embedding": cls_embedding_full,
            "contrastive_embedding": contrastive_embedding_full,
            "multiview_embeddings": multiview_embeddings,
        }

    def _build_sub_x_list_random(self, x: pd.DataFrame, num_sub_cols: List[int]):
        cols = x.columns.tolist()
        total_cols = len(cols)

        if total_cols != num_sub_cols[0]:
            raise ValueError(f"expect {num_sub_cols[0]} columns, get {total_cols} instead")

        sub_x_list = []
        for count in num_sub_cols:
            if count == total_cols:
                selected_cols = cols
            else:
                indices = np.random.choice(total_cols, count, replace=False)
                selected_cols = [cols[i] for i in indices]
            sub_x = x.copy()[selected_cols]
            sub_x_list.append(sub_x)

        return sub_x_list

    def save(self, ckpt_dir):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)

        torch.save(self.state_dict(), os.path.join(ckpt_dir, constants.WEIGHTS_NAME))

        if self.input_encoder.feature_extractor is not None:
            self.input_encoder.feature_extractor.save(ckpt_dir)

        model_params = {
            "categorical_columns": self.input_encoder.feature_extractor.categorical_columns,
            "numerical_columns": self.input_encoder.feature_extractor.numerical_columns,
            "binary_columns": self.input_encoder.feature_extractor.binary_columns,
            "hidden_dim": self.encoder.hidden_dim,
            "num_layer": self.encoder.num_layer,
            "num_attention_head": self.encoder.num_attention_head,
            "hidden_dropout_prob": self.encoder.hidden_dropout_prob,
            "ffn_dim": self.encoder.ffn_dim,
            "num_sub_cols": self.num_sub_cols,
            "activation": self.activation,
            "separate_contrast_token": self.separate_contrast_token,
        }

        with open(os.path.join(ckpt_dir, constants.TRANSTAB_PARAMS_NAME), "w") as f:
            json.dump(model_params, f, indent=4)

        state_dict_input_encoder = self.input_encoder.state_dict()
        torch.save(state_dict_input_encoder, os.path.join(ckpt_dir, constants.INPUT_ENCODER_NAME))