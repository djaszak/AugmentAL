from core.constants import AugmentationMethods, Datasets
from script import run_script

run_script(
    AugmentationMethods.BACK_TRANSLATION.value,
    query_strategies=["AugmentedSearchSpaceExtensionQueryStrategy",],
    dataset=Datasets.TWEET.value
)