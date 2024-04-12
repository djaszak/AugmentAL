from core.loop import run_active_learning_loop
from core.constants import Datasets, TransformerModels, AugmentationMethods
from datasets import load_dataset
from core.augment import create_augmented_dataset

query_strategies = [
    # "RandomSampling",
    # "BreakingTies",
    # "AugmentedSearchSpaceExtensionQueryStrategy",
    # "AugmentedSearchSpaceExtensionAndOutcomeQueryStrategy",
    # "AugmentedOutcomesQueryStrategy",
    "AverageAcrossAugmentedQueryStrategy",
    # "AverageAcrossAugmentedExtendedOutcomesQueryStrategy",
]

num_queries = 5
num_samples = 2
num_augmentations = 2

datasets = [Datasets.ROTTEN.value]

for dataset_name in datasets:

    dataset = load_dataset(dataset_name)

    raw_test = dataset["test"]
    raw_train = dataset["train"]
    raw_augmented_train, augmented_indices = create_augmented_dataset(
        raw_train, AugmentationMethods.BART_SUBSTITUTE.value, n=num_augmentations
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
            query_strategy=query_strategy,
            model=TransformerModels.BERT.value,
            device="cuda",
        )
