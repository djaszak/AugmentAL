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

