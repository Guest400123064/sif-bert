from typing import Dict, Any
import logging

import os
import json

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class BatchPCRemoval(nn.Module):
    """Inspired by the Smooth Inverse Frequency (SIF) common direction removal 
        method in https://openreview.net/pdf?id=SyK00v5xx. This layer is NOT trainable.
        Instead of learning the common direction, it performs a in-batch estimation by 
        calculating the SVD of the sentence embeddings and removing the first `n_components`.
        Therefore, larger batch sizes will result in better estimation, i.e., closer to 
        SIF, which operates over the entire dataset."""
    
    def __init__(self, n_components: int = 1):
        """Set number of components to remove from sentence embeddings.
        
        :param n_components: Number of components to remove from sentence embeddings.
        :type n_components: int
        """
        super(BatchPCRemoval, self).__init__()
        self.config_keys = ["n_components"]
        self.n_components = n_components

    def forward(self, features: Dict[str, Any]):
        """Given a batch of sentence embeddings, remove the first `n_components`.
            To calculate the common direction, we need to `.detach()` the embeddings
            so that the SVD does not back-propagate to the embeddings."""

        embeddings = features["sentence_embedding"]
        embeddings_detached = embeddings.detach()

        _, _, components = torch.linalg.svd(embeddings_detached, full_matrices=False)
        components = components[:self.n_components]
        projection = torch.matmul(embeddings_detached, components.transpose(1, 0))

        features.update({"sentence_embedding": embeddings - torch.matmul(projection, components)})
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
