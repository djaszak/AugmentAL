import os
import json
import pandas as pd

from datetime import datetime
from pathlib import Path

from core.augment import create_augmented_dataset
from core.constants import AugmentationMethods, Datasets, TransformerModels
from core.loop import run_active_learning_loop
from datasets import load_dataset, load_from_disk


num_queries = 50
num_samples = 20
num_augmentations = 5
chosen_dataset = Datasets.IMDB.value
# datasets_path = "/beegfs/ws/1/s8822750-active-learning-data-augmentation/datasets"
datasets_path = "/Users/dennis/Library/Mobile Documents/com~apple~CloudDocs/Uni/DiplomArbeit/datasets"


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
    if augmentation_method:
        # Try to load a local already augmented dataset if it exists.
        potential_training_set_path = f"{datasets_path}/{dataset_name}/{augmentation_method}"
        if os.path.exists(potential_training_set_path):
            dataset = load_from_disk(potential_training_set_path)
            augmented_indices = json.load(f"{potential_training_set_path}/augmented_indices.json")

            raw_train = dataset["train"]
            raw_test = dataset["test"]
        else:
            dataset = load_dataset(dataset_name)

            raw_train = dataset["train"]
            raw_test = dataset["test"]
            raw_train, augmented_indices = create_augmented_dataset(
                raw_train, augmentation_method, n=num_augmentations
            )
    else:
        augmented_indices = {}
        dataset = load_dataset(dataset_name)

        raw_test = dataset["test"]
        raw_train = dataset["train"]
    
    return raw_test, raw_train, augmented_indices


def run_script(
    augmentation_method: AugmentationMethods | None = None, repetitions: int = 5
):
    start_time = datetime.now()
    print(f"Starting run at {start_time}. \n")
    path = (
        Path(__file__).parent / "results" / str(augmentation_method).replace(" ", "_")
    ).resolve()

    try:
        os.mkdir(path)
        print(f"Directory {path} successfully created.")
    except OSError:
        print(f"{path} already exists.")

    raw_test, raw_train, augmented_indices = create_raw_set(
        chosen_dataset, augmentation_method
    )

    query_strategies = (
        [
            # Basic Augmented Strategies
            "AugmentedSearchSpaceExtensionQueryStrategy",
            "AugmentedOutcomesQueryStrategy",
            "AverageAcrossAugmentedQueryStrategy",
            # Combinations
            "AugmentedSearchSpaceExtensionAndOutcomeQueryStrategy",
            "AverageAcrossAugmentedExtendedOutcomesQueryStrategy",
        ]
        if augmentation_method
        else [
            # Basic Strategies
            "RandomSampling",
            "BreakingTies",
        ]
    )

    for query_strategy in query_strategies:
        final_results = {}

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

            with open((path / saving_name).resolve(), "w") as f:
                f.write(pd.DataFrame(final_results).to_json())

    end_time = datetime.now()
    print(f"Finished run at {end_time}. \n")
    print(f"Run took {end_time - start_time}. \n")
