from core.loop import run_active_learning_loop
from core.constants import Datasets
from datasets import load_dataset
from nlpaug.augmenter import word as naw
from core.augment import create_augmented_dataset
from enum import Enum

# run_active_learning_loop(
#     num_queries=3,
#     num_samples=20,
#     num_augmentations=1,
#     dataset=Datasets.ROTTEN,
#     query_strategy="AugmentedSearchSpaceExtensionQueryStrategy",
# )

query_strategies = [
    # "BreakingTies",
    # "AugmentedSearchSpaceExtensionAndOutcomeQueryStrategy",
    # "AugmentedSearchSpaceExtensionQueryStrategy",
    # "AugmentedOutcomesQueryStrategy",
    # "AugmentedSearchSpaceExtensionAndOutcomeLeastConfidenceQueryStrategy",
    # "AugmentedSearchSpaceExtensionLeastConfidenceQueryStrategy",
    "AugmentedOutcomesLeastConfidenceQueryStrategy",
]

num_queries = 20
num_samples = 20
num_augmentations = 5

datasets = [Datasets.ROTTEN.value]

for dataset_name in datasets:

    dataset = load_dataset(dataset_name)

    raw_test = dataset["test"]
    raw_train = dataset["train"]
    raw_augmented_train, augmented_indices = create_augmented_dataset(
        raw_train, naw.SynonymAug(aug_src="wordnet"), n=num_augmentations
    )

    for query_strategy in query_strategies:
        run_active_learning_loop(
            raw_test,
            raw_train,
            raw_augmented_train,
            augmented_indices,
            num_queries=num_queries,
            num_samples=num_samples,
            num_augmentations=num_augmentations
            if query_strategy != "BreakingTies"
            else 0,
            dataset=dataset,
            query_strategy=query_strategy,
        )
