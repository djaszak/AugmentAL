import os
import pandas as pd

from pathlib import Path

from core.augment import create_augmented_dataset
from core.constants import AugmentationMethods, Datasets, TransformerModels
from core.loop import run_active_learning_loop
from datasets import load_dataset

query_strategies = [
    # Basic Strategies
    "RandomSampling",
    "BreakingTies",
    # Basic Augmented Strategies
    "AugmentedSearchSpaceExtensionQueryStrategy",
    "AugmentedOutcomesQueryStrategy",
    "AverageAcrossAugmentedQueryStrategy",
    # Combinations
    "AugmentedSearchSpaceExtensionAndOutcomeQueryStrategy",
    "AverageAcrossAugmentedExtendedOutcomesQueryStrategy",
]

datasets = [Datasets.ROTTEN.value]

num_queries = 50
num_samples = 20
num_augmentations = 5


def create_raw_set(
    dataset_name: str, augmentation_method: AugmentationMethods | None = None
):
    """If augmentation_method is not None, create an augmented dataset

    Args:
        dataset_name (str): Name of the dataset, string has to be a name, that is represented at huggingface.com/datasets
        augmentation_method (AugmentationMethods | None, optional): The augmentation method, available ones can be found at given enum. Defaults to None.

    Returns:
        tuple(dataset, dataset, dict): Raw sets and augmented indices, if augmentation_method is not None, else just the sets and an empty dict.
    """
    dataset = load_dataset(dataset_name)

    raw_test = dataset["test"]
    raw_train = dataset["train"]
    augmented_indices = {}
    if augmentation_method:
        raw_train, augmented_indices = create_augmented_dataset(
            raw_train, augmentation_method, n=num_augmentations
        )

    return raw_test, raw_train, augmented_indices


def run_script(augmentation_method: AugmentationMethods | None = None, repetitions: int = 5):
    path = (Path(__file__).parent / "../results" / augmentation_method.replace(", ", "_")).resolve()

    try:
        os.mkdir(path)
        print(f"Directory {} successfully created.")
    except OSError as error:
        print(f"{path} already exists.")

    raw_augmented_test, raw_train, augmented_indices = create_raw_set(
                dataset_name, augmentation_method
            ) 

    raw_unaugmented_test, raw_train, augmented_indices = create_raw_set(
        dataset_name
    )

    for query_strategy in query_strategies:
        final_results = {}
        for dataset_name in datasets:
            raw_test = raw_unaugmented_test if query_strategy == "BreakingTies" or query_strategy == "RandomSampling" else raw_augmented_test

            for rep in range(repetitions):
                results = run_active_learning_loop(
                    raw_test,
                    raw_train,
                    augmented_indices,
                    num_queries=num_queries,
                    num_samples=num_samples,
                    num_augmentations=num_augmentations
                    if query_strategy != "BreakingTies"
                    or query_strategy != "RandomSampling"
                    else 0,
                    query_strategy=query_strategy,
                    model=TransformerModels.BERT.value,
                    device="cuda",
                )

                final_results[rep] = results
                saving_name = f"{query_strategy}_{num_queries}_queries_num_samples_{num_samples}_num_augmentations_{num_augmentations}.json"

                with open((path / saving_name).resolve(), "wb") as f:
                    f.write(pd.DataFrame(final_results).to_json())
