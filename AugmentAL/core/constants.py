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


class AugmentationMethods(Enum):
    RANDOM_SWAP = nac.RandomCharAug(action="swap")
    RANDOM_INSERT = nac.RandomCharAug(action="insert")
    RANDOM_SUBSTITUTE = nac.RandomCharAug(action="substitute")
    RANDOM_DELETE = nac.RandomCharAug(action="delete")
    SYNONYM = naw.SynonymAug(aug_src="wordnet")
    BART_SUBSTITUTE = naw.ContextualWordEmbsAug(
        model_path="facebook/bart-base",
        model_type="bart",
        action="substitute",
        use_custom_api=False,
        device="cuda",
    )
    BERT_SUBSTITUTE = naw.ContextualWordEmbsAug(
        model_path="bert-base-uncased", action="substitute"
    )
    BACK_TRANSLATION = naw.BackTranslationAug(
        from_model_name="facebook/wmt19-en-de",
        to_model_name="facebook/wmt19-de-en",
        device="cuda",
    )
    GENERATIVE_GPT2 = nas.ContextualWordEmbsForSentenceAug(
        model_path="gpt2", device="cuda"
    )
    GENERATIVE_DISTILGPT2 = nas.ContextualWordEmbsForSentenceAug(
        model_path="distilgpt2", device="cuda"
    )
    ABSTRACTIVE_SUMMARIZATION = nas.AbstSummAug(model_path="t5-base", device="cuda")
