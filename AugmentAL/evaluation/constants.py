import enum


class Datasets(enum.Enum):
    ROTTEN = "rotten_tomatoes"
    IMDB = "imdb"
    AG_NEWS = "fancyzhx/ag_news"
    TWEET = "tweet_eval"


class AugmentationType(enum.Enum):
    BACK_TRANSLATION_AUG = "Name:BackTranslationAug,_Action:substitute,_Method:word"
    SYNONYM_AUG = "Name:Synonym_Aug,_Aug_Src:wordnet,_Action:substitute,_Method:word"
    CONTEXTUAL_WORD_EMBS_FOR_SENTENCE = (
        "Name:ContextualWordEmbsForSentence_Aug,_Action:insert,_Method:sentence"
    )
    CONTEXTUAL_WORD_EMBS = "Name:ContextualWordEmbs_Aug,_Action:substitute,_Method:word"
    RANDOM_SWAP = "Name:RandomChar_Aug,_Action:swap,_Method:char"

class FolderPaths(enum.Enum):
    NO_AUG_TWEET = f"None/{Datasets.TWEET.value}"
    NO_AUG_IMDB = f"None/{Datasets.IMDB.value}"
    NO_AUG_AG_NEWS = f"None/{Datasets.AG_NEWS.value}"
    BACKTRANSLATION_TWEET = f"{AugmentationType.BACK_TRANSLATION_AUG.value}/{Datasets.TWEET.value}"
    BACKTRANSLATION_IMDB = f"{AugmentationType.BACK_TRANSLATION_AUG.value}/{Datasets.IMDB.value}"
    BACKTRANSLATION_AG_NEWS = f"{AugmentationType.BACK_TRANSLATION_AUG.value}/{Datasets.AG_NEWS.value}"
    SYNONYM_TWEET = f"{AugmentationType.SYNONYM_AUG.value}/{Datasets.TWEET.value}"
    SYNONYM_IMDB = f"{AugmentationType.SYNONYM_AUG.value}/{Datasets.IMDB.value}"
    SYNONYM_AG_NEWS = f"{AugmentationType.SYNONYM_AUG.value}/{Datasets.AG_NEWS.value}"
    BERT_TWEET = f"{AugmentationType.CONTEXTUAL_WORD_EMBS.value}/{Datasets.TWEET.value}"
    BERT_IMDB = f"{AugmentationType.CONTEXTUAL_WORD_EMBS.value}/{Datasets.IMDB.value}"
    BERT_AG_NEWS = f"{AugmentationType.CONTEXTUAL_WORD_EMBS.value}/{Datasets.AG_NEWS.value}"
    RANDOM_SWAP_TWEET = f"{AugmentationType.RANDOM_SWAP.value}/{Datasets.TWEET.value}"
    RANDOM_SWAP_IMDB = f"{AugmentationType.RANDOM_SWAP.value}/{Datasets.IMDB.value}"
    RANDOM_SWAP_AG_NEWS = f"{AugmentationType.RANDOM_SWAP.value}/{Datasets.AG_NEWS.value}"


class QueryStrategy(enum.Enum):
    RANDOM_SAMPLING = "RandomSampling"
    BREAKING_TIES = "BreakingTies"


LATEX_IMAGES_PATH = "/Users/dennis/Library/Mobile Documents/com~apple~CloudDocs/Uni/DiplomArbeit/DiplomLatex/images/"

QUERY_STRATEGY_COLUMN = "query_strategy"

STOPPING_CRITERIA = [
    "kappa_average_conservative_history",
    "kappa_average_middle_ground_history",
    "kappa_average_aggressive_history",
    "delta_f_score_conservative_history",
    "delta_f_score_middle_ground_history",
    "delta_f_score_aggressive_history",
    "classification_change_conservative_history",
    "classification_change_middle_ground_history",
    "classification_change_aggressive_history",
]

