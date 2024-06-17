from core.constants import AugmentationMethods, Datasets
from script import run_script

run_script(
    AugmentationMethods.BERT_SUBSTITUTE.value,
    query_strategies=[
        "AverageAcrossAugmentedQueryStrategy",
    ],
    dataset=Datasets.TWEET.value,
)
