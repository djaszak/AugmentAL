from core.constants import AugmentationMethods, Datasets
from script import run_script

run_script(
    AugmentationMethods.RANDOM_SWAP.value,
    query_strategies=[
        "AugmentedSearchSpaceExtensionQueryStrategy",
    ],
    dataset=Datasets.IMDB.value,
)
