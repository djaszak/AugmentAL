from core.constants import AugmentationMethods, Datasets
from script import run_script

run_script(
    AugmentationMethods.SYNONYM.value,
    query_strategies=["AverageAcrossAugmentedQueryStrategy",],
    dataset=Datasets.TWEET.value
)