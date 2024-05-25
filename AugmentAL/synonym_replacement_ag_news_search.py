from core.constants import AugmentationMethods, Datasets
from script import run_script

run_script(
    AugmentationMethods.SYNONYM.value,
    query_strategies=["AugmentedSearchSpaceExtensionQueryStrategy",],
    dataset=Datasets.AG_NEWS.value
)