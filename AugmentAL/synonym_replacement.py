from core.constants import AugmentationMethods
from script import run_script

run_script(AugmentationMethods.SYNONYM.value, query_strategies=["AugmentedOutcomesQueryStrategy",
                "AverageAcrossAugmentedQueryStrategy",])
