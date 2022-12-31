from typing import Dict, Any
import logging

import os
import json

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class MeanPooling(nn.Module):
    """Implements simple mean pooling of the token embeddings."""
        
    def __init__(self):
        super(MeanPooling, self).__init__()
        self.config_keys = []
    
    def forward(self, features: Dict[str, Any]):
        """Performs mean pooling on the token embeddings."""

        embeddings  = features["token_embeddings"].transpose(1, 2)
        attn_masks  = features["attention_mask"].unsqueeze(-1).float()
        
        seq_lengths = torch.clamp(attn_masks.sum(1, keepdim=True), min=1e-9)
        weights = (attn_masks / seq_lengths)
        
        features.update({"sentence_embedding": torch.bmm(embeddings, weights).squeeze(-1)})
        return features
    
    # For IO //////////////////////////////////////////////////////////////////////
    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(self.get_config_dict(), f, indent=2)

    @classmethod
    def load(cls, input_path):
        with open(os.path.join(input_path, "config.json")) as f:
            config = json.load(f)
        return cls(**config)
