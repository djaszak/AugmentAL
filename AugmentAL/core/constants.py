from enum import Enum

# TODO: Add query strategies as from loop.py as an Enum


class TransformerModels(Enum):
    BERT_TINY = "prajjwal1/bert-tiny"


class Datasets(Enum):
    ROTTEN = "rotten_tomatoes"
