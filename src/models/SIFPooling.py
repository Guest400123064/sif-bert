from typing import List, Dict, Union, Any
import logging

import os
import json

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class SIFPooling(nn.Module):
    """Implements the Smooth Inverse Frequency (SIF) weighting scheme
        mentioned in https://openreview.net/pdf?id=SyK00v5xx. This 
        layer is NOT trainable."""
        
    def __init__(self, 
                 word_embedding_dimension: int,
                 vocab:        List[str],
                 token_counts: Dict[str, int], 
                 a:            float = 1e-3):
        """Initializes the SIF layer.
        
        :param word_embedding_dimension: Output sentence embedding dimension.
        :type word_embedding_dimension: int

        :param vocab: The vocabulary of the model. MUST be the same as the
            vocabulary used for base model. Note that the order of tokens 
            in the vocabulary should also be aligned since the SIF weights 
            will be indexed by `input_ids`.
        :type vocab: List[str]
        
        :param token_counts: A dictionary mapping tokens to their counts in
            the corpus. The keys of the dictionary should be a subset of `vocab`.
        :type token_counts: Dict[str, int]    
        
        :param a: The `a` parameter of the SIF weighting scheme.
        :type a: float
        """
        super(SIFPooling, self).__init__()
        self.config_keys  = ["word_embedding_dimension", "vocab", "token_counts", "a"]
        self.word_embedding_dimension = word_embedding_dimension
        self.vocab        = vocab
        self.token_counts = token_counts
        self.a            = a
        
        # The calculation of the weights is slightly different from the 
        # original paper for numerical stability. We first down scale the
        # counts by the `a` parameter and then compute the weights.
        # Original version:
        #   wi = a / (a + pi) = a / (a + ci / N)
        # This implementation:
        #   wi = 1 / (1 + ci / (a * N)) = 1 / (1 + ci / denom)
        denom = sum(a * c for c in token_counts.values())
        
        weights = torch.zeros((len(vocab), 1), dtype=torch.float)
        num_unknown_words = 0
        for i, token in enumerate(vocab):
            count = token_counts.get(token, 0)
            num_unknown_words += int(count == 0)
            weights[i] = (1 / (1 + count / denom))
        
        logger.info(f"{num_unknown_words} of {len(vocab)} words without a "
                        "`count` value; setting SIF weight to 1.0")
        self.emb_layer = nn.Embedding.from_pretrained(weights, freeze=True)
    
    def forward(self, features: Dict[str, Any]):
        """Performs SIF weighting by multiplying the token embeddings with the
            corresponding SIF weights. The resulting token embeddings are then
            averaged by length to obtain the sentence embedding."""
            
        embeddings  = features["token_embeddings"].transpose(1, 2)
        attn_masks  = features["attention_mask"].unsqueeze(-1).float()
        
        seq_lengths = torch.clamp(attn_masks.sum(1, keepdim=True), min=1e-9)
        sif_weights = (self.emb_layer(features["input_ids"]) * attn_masks / seq_lengths)
        
        features.update({"sentence_embedding": torch.bmm(embeddings, sif_weights).squeeze(-1)})
        return features
    
    @classmethod
    def from_corpus_hf(cls, 
                       word_embedding_dimension: int, 
                       model_card: str, 
                       corpus: Union[str, List[str]], 
                       a: float = 1e-3) -> "SIFPooling":
        """Estimates the word frequencies (actually word counts) from a corpus of strings. 
            The sentences in the corpus are tokenized using the corresponding 
            Huggingface transformers tokenizer of the model for token frequency estimation.
        
        :param word_embedding_dimension: Output sentence embedding dimension.
        :type word_embedding_dimension: int
        
        :param model_card: The model card of the model to use for tokenization, e.g., `bert-base-uncased`.
        :type model_card: str
        
        :param corpus: String, or list of strings, to use for token frequency estimation.
        :type corpus: Union[str, List[str]]
        
        :param a: The `a` parameter of the SIF weighting scheme.
        :type a: float
        """
        
        from collections import Counter
        from transformers import AutoTokenizer
        
        if isinstance(corpus, str):
            corpus = [corpus]
        
        tokenizer = AutoTokenizer.from_pretrained(model_card)
        token_counts = Counter()
        for s in corpus:
            tokens = tokenizer.tokenize(s, add_special_tokens=True, truncation=True)
            token_counts.update(tokens)
        
        # In order to make sure that the weightings (using `nn.Embedding`) aligns 
        # with the token ids, we need to sort the tokens by their ids. 
        vocab = [k for k, _ in sorted(tokenizer.get_vocab().items(), key=lambda x: x[1])]
        return cls(word_embedding_dimension, vocab, token_counts, a=a)
    
    # For IO //////////////////////////////////////////////////////////////////////
    def get_sentence_embedding_dimension(self):
        return self.word_embedding_dimension

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
