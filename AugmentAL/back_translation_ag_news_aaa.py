print("STARTING")
from core.constants import AugmentationMethods, Datasets
from script import run_script

print("IMPORTS LOADED")

run_script(
    AugmentationMethods.BACK_TRANSLATION.value,
    query_strategies=[
        "AverageAcrossAugmentedQueryStrategy",
    ],
    dataset=Datasets.AG_NEWS.value,
)
