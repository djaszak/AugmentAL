print("IN CONSTANTS")
from enum import Enum

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.word as naw

# TODO: Add query strategies as from loop.py as an Enum


class TransformerModels(Enum):
    BERT_TINY = "prajjwal1/bert-tiny"
    BERT = "google-bert/bert-base-uncased"


class Datasets(Enum):
    ROTTEN = "rotten_tomatoes"
    IMDB = "imdb"
    AG_NEWS = "fancyzhx/ag_news"
    TWEET = "tweet_eval"


class AugmentationMethods(Enum):
    RANDOM_SWAP = "Name:RandomChar_Aug, Action:swap, Method:char"
    SYNONYM = "Name:Synonym_Aug, Aug Src:wordnet, Action:substitute, Method:word"
    BERT_SUBSTITUTE = "Name:ContextualWordEmbs_Aug, Action:substitute, Method:word"
    BACK_TRANSLATION = "Name:BackTranslationAug, Action:substitute, Method:word"


# class AugmentationMethods(Enum):
#     RANDOM_SWAP = nac.RandomCharAug(action="swap")
#     # RANDOM_INSERT = nac.RandomCharAug(action="insert")
#     # RANDOM_SUBSTITUTE = nac.RandomCharAug(action="substitute")
#     # RANDOM_DELETE = nac.RandomCharAug(action="delete")
#     SYNONYM = naw.SynonymAug(aug_src="wordnet")
#     # BART_SUBSTITUTE = naw.ContextualWordEmbsAug(
#         # model_path="facebook/bart-base",
#         # model_type="bart",
#         # action="substitute",
#         # use_custom_api=False,
#         # device="cuda",
#     # )
#     BERT_SUBSTITUTE = naw.ContextualWordEmbsAug(
#         model_path="bert-base-uncased", action="substitute",
#         # force_reload=True,
#     )
#     BACK_TRANSLATION = naw.BackTranslationAug(
#         from_model_name="facebook/wmt19-en-de",
#         to_model_name="facebook/wmt19-de-en",
#         device="cuda",
#         # force_reload=True,
#     )
#     # GENERATIVE_GPT2 = nas.ContextualWordEmbsForSentenceAug(
#         # model_path="gpt2", device="cuda"
#     # )
#     # GENERATIVE_DISTILGPT2 = nas.ContextualWordEmbsForSentenceAug(
#         # model_path="distilgpt2", device="cuda"
#     # )
#     # ABSTRACTIVE_SUMMARIZATION = nas.AbstSummAug(model_path="t5-base", device="cuda")
