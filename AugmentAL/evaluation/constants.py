import enum


class Datasets(enum.Enum):
    # ROTTEN = "rotten_tomatoes"
    IMDB = "imdb"
    AG_NEWS = "fancyzhx/ag_news"
    TWEET = "tweet_eval"


class AugmentationType(enum.Enum):
    BACK_TRANSLATION_AUG = "Name:BackTranslationAug,_Action:substitute,_Method:word"
    SYNONYM_AUG = "Name:Synonym_Aug,_Aug_Src:wordnet,_Action:substitute,_Method:word"
    # CONTEXTUAL_WORD_EMBS_FOR_SENTENCE = (
    #     "Name:ContextualWordEmbsForSentence_Aug,_Action:insert,_Method:sentence"
    # )
    CONTEXTUAL_WORD_EMBS = "Name:ContextualWordEmbs_Aug,_Action:substitute,_Method:word"
    RANDOM_SWAP = "Name:RandomChar_Aug,_Action:swap,_Method:char"


class BasePaths(enum.Enum):
    NO_AUG_TWEET = f"None/{Datasets.TWEET.value}"
    NO_AUG_IMDB = f"None/{Datasets.IMDB.value}"
    NO_AUG_AG_NEWS = f"None/{Datasets.AG_NEWS.value}"


class AugmentedPaths(enum.Enum):
    SYNONYM_TWEET = f"{AugmentationType.SYNONYM_AUG.value}/{Datasets.TWEET.value}"
    SYNONYM_IMDB = f"{AugmentationType.SYNONYM_AUG.value}/{Datasets.IMDB.value}"
    SYNONYM_AG_NEWS = f"{AugmentationType.SYNONYM_AUG.value}/{Datasets.AG_NEWS.value}"
    BERT_TWEET = f"{AugmentationType.CONTEXTUAL_WORD_EMBS.value}/{Datasets.TWEET.value}"
    BERT_IMDB = f"{AugmentationType.CONTEXTUAL_WORD_EMBS.value}/{Datasets.IMDB.value}"
    BERT_AG_NEWS = (
        f"{AugmentationType.CONTEXTUAL_WORD_EMBS.value}/{Datasets.AG_NEWS.value}"
    )
    RANDOM_SWAP_TWEET = f"{AugmentationType.RANDOM_SWAP.value}/{Datasets.TWEET.value}"
    RANDOM_SWAP_IMDB = f"{AugmentationType.RANDOM_SWAP.value}/{Datasets.IMDB.value}"
    RANDOM_SWAP_AG_NEWS = (
        f"{AugmentationType.RANDOM_SWAP.value}/{Datasets.AG_NEWS.value}"
    )
    BACKTRANSLATION_TWEET = (
        f"{AugmentationType.BACK_TRANSLATION_AUG.value}/{Datasets.TWEET.value}"
    )
    BACKTRANSLATION_IMDB = (
        f"{AugmentationType.BACK_TRANSLATION_AUG.value}/{Datasets.IMDB.value}"
    )
    BACKTRANSLATION_AG_NEWS = (
        f"{AugmentationType.BACK_TRANSLATION_AUG.value}/{Datasets.AG_NEWS.value}"
    )


class AugmentedStrategies(enum.Enum):
    AUGMENTED_OUTCOME = "AugmentedOutcomesQueryStrategy"
    AUGMENTED_SEARCH_SPACE = "AugmentedSearchSpaceExtensionQueryStrategy"
    AVERAGE_ACROSS_AUGMENTED = "AverageAcrossAugmentedQueryStrategy"


class BaselineStrategies(enum.Enum):
    RANDOM_SAMPLING = "RandomSampling"
    BREAKING_TIES = "BreakingTies"


QUERY_STRATEGIES_VERBOSE = {
    BaselineStrategies.RANDOM_SAMPLING.value: "Random Sampling",
    BaselineStrategies.BREAKING_TIES.value: "Breaking Ties",
    AugmentedStrategies.AUGMENTED_OUTCOME.value: "Extended Outcome",
    AugmentedStrategies.AUGMENTED_SEARCH_SPACE.value: "Extended Search Space",
    AugmentedStrategies.AVERAGE_ACROSS_AUGMENTED.value: "AAA",
}

AUGMENTATION_METHOD_VERBOSE = {
    AugmentationType.BACK_TRANSLATION_AUG.value: "Backtranslation",
    AugmentationType.SYNONYM_AUG.value: "Synonym",
    AugmentationType.CONTEXTUAL_WORD_EMBS.value: "BERT",
    AugmentationType.RANDOM_SWAP.value: "Random Swap",
}

DATASETS_VERBOSE = {
    Datasets.IMDB.value: "IMDB",
    Datasets.AG_NEWS.value: "AG News",
    Datasets.TWEET.value: "Tweet",
}

DATASETS_VALUES_VERBOSE = {
    Datasets.IMDB.value: "IMDB",
    Datasets.AG_NEWS.value: "AG News",
    Datasets.TWEET.value: "Tweet",
}

STOPPING_CRITERIA_VERBOSE = {
    "kappa_average_conservative_history": "Kappa Conservative",
    "kappa_average_middle_ground_history": "Kappa Middle Ground",
    "kappa_average_aggressive_history": "Kappa Aggressive",
    "delta_f_score_conservative_history": "Delta F-Score Conservative",
    "delta_f_score_middle_ground_history": "Delta F-Score Middle Ground",
    "delta_f_score_aggressive_history": "Delta F-Score Aggressive",
    "classification_change_conservative_history": "Classification Change Conservative",
    "classification_change_middle_ground_history": "Classification Change Middle Ground",
    "classification_change_aggressive_history": "Classification Change Aggressive",
}

STOPPING_CRITERIA_VERBOSE_SHORT = {
    "kappa_average_conservative_history": "KA Con",
    "kappa_average_middle_ground_history": "KA MG",
    "kappa_average_aggressive_history": "KA Aggr",
    "delta_f_score_conservative_history": "DF Con",
    "delta_f_score_middle_ground_history": "DF MG",
    "delta_f_score_aggressive_history": "DF Aggr",
    "classification_change_conservative_history": "CC Con",
    "classification_change_middle_ground_history": "CC MG",
    "classification_change_aggressive_history": "CC Aggr",
}

LATEX_TABLES_PATH = "/Users/dennis/Library/Mobile Documents/com~apple~CloudDocs/Uni/DiplomArbeit/DiplomLatex/tables/"
LATEX_TABLES_VARIANCE_PATH = "/Users/dennis/Library/Mobile Documents/com~apple~CloudDocs/Uni/DiplomArbeit/DiplomLatex/tables_variance/"
LATEX_TABLES_STOP_PATH = "/Users/dennis/Library/Mobile Documents/com~apple~CloudDocs/Uni/DiplomArbeit/DiplomLatex/tables_stop/"
LATEX_TABLES_RAW_PATH = "/Users/dennis/Library/Mobile Documents/com~apple~CloudDocs/Uni/DiplomArbeit/DiplomLatex/tables_raw/"
LATEX_IMAGES_PATH = "/Users/dennis/Library/Mobile Documents/com~apple~CloudDocs/Uni/DiplomArbeit/DiplomLatex/images/"

QUERY_STRATEGY_COLUMN = "query_strategy"
DATASET_COLUMN = "dataset"
AUGMENTATION_METHOD_COLUMN = "augmentation_method"

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

AUGMENTATION_METHODS_DICT = {
    AugmentedPaths.BACKTRANSLATION_TWEET.value: "Backtranslation",
    AugmentedPaths.SYNONYM_TWEET.value: "Synonym",
    AugmentedPaths.BERT_TWEET.value: "BERT",
    AugmentedPaths.RANDOM_SWAP_TWEET.value: "Random Swap",
    AugmentedPaths.BACKTRANSLATION_IMDB.value: "Backtranslation",
    AugmentedPaths.SYNONYM_IMDB.value: "Synonym",
    AugmentedPaths.BERT_IMDB.value: "BERT",
    AugmentedPaths.RANDOM_SWAP_IMDB.value: "Random Swap",
    AugmentedPaths.BACKTRANSLATION_AG_NEWS.value: "Backtranslation",
    AugmentedPaths.SYNONYM_AG_NEWS.value: "Synonym",
    AugmentedPaths.BERT_AG_NEWS.value: "BERT",
    AugmentedPaths.RANDOM_SWAP_AG_NEWS.value: "Random Swap",
}
