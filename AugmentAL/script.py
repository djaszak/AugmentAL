from core.constants import Dataset
from core.loop import run_active_learning_loop

datasets = list(Dataset)
query_strategies = [
    "BreakingTies",
    "AugmentedSearchSpaceExtensionAndOutcomeQueryStrategy",
    "AugmentedSearchSpaceExtensionQueryStrategy",
    "AugmentedOutcomesQueryStrategy",
    "AugmentedSearchSpaceExtensionAndOutcomeLeastConfidenceQueryStrategy",
    "AugmentedSearchSpaceExtensionLeastConfidenceQueryStrategy",
    "AugmentedOutcomesLeastConfidenceQueryStrategy",
]

num_queries = 20
num_samples = 20
num_augmentations = 5

for dataset in datasets:
    for query_strategy in query_strategies:
        run_active_learning_loop(
            num_queries=num_queries,
            num_samples=num_samples,
            num_augmentations=num_augmentations
            if query_strategy != "BreakingTies"
            else 0,
            dataset=dataset,
            query_strategy=query_strategy,
        )
