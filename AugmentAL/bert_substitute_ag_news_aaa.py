print("LETS REALLY GET STARTED, THIS IS THE FIRST STARTING POINT")
from core.constants import AugmentationMethods, Datasets
print("LOADED CONSTANTS")
from script import run_script
print("ALL IMPORTS DONE")

run_script(
    AugmentationMethods.BERT_SUBSTITUTE.value,
    query_strategies=["AverageAcrossAugmentedQueryStrategy",],
    dataset=Datasets.AG_NEWS.value
)
