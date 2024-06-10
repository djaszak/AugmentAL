print("LETS GET STARTED")
import os
import json
import pandas as pd

from datetime import datetime
from pathlib import Path

from core.augment import create_augmented_dataset
from core.constants import AugmentationMethods, Datasets, TransformerModels
from core.loop import run_active_learning_loop
from datasets import load_dataset, load_from_disk
from torch.multiprocessing import set_start_method
import multiprocessing as mp

# mp.set_start_method('spawn')
# print("START METHOD WAS SET WITH MP LIB")


# try:
#     print("SETTING START METHOD") 
#     set_start_method('spawn', force=True)
#     print("START METHOD SET")
# except RuntimeError as e:
#     print(f"START METHOD SETTIN DID NOT WORK WITH ERROR {e}")


num_queries = 50
num_samples = 20
num_augmentations = 5
chosen_dataset = Datasets.IMDB.value
# datasets_path = "/data/horse/ws/s8822750-active-learning-data-augmentation/datasets"
# datasets_path = "/Users/dennis/Library/Mobile Documents/com~apple~CloudDocs/Uni/DiplomArbeit/datasets"


def create_raw_set(
    dataset_name: str, augmentation_method: AugmentationMethods | None = None, saving_path: str = "/data/horse/ws/s8822750-active-learning-data-augmentation/datasets"
):
    """If augmentation_method is not None, create an augmented dataset

    Args:
        dataset_name (str): Name of the dataset, string has to be a name, that is represented at huggingface.com/datasets
        augmentation_method (AugmentationMethods | None, optional): The augmentation method, available ones can be found at given enum. Defaults to None.

    Returns:
        tuple(dataset, dataset, dict): Raw sets and augmented indices, if augmentation_method is not None, else just the sets and an empty dict.
    """
    if "tweet" in dataset_name:
        loaded_dataset = load_dataset(dataset_name, "irony")
    else:
        loaded_dataset = load_dataset(dataset_name)
    augmented_indices = {}
    if augmentation_method:
        # Try to load a local already augmented dataset if it exists.
        potential_training_set_path = f"{saving_path}/{dataset_name}/{augmentation_method}"
        print(potential_training_set_path)
        if os.path.exists(potential_training_set_path) and os.listdir(potential_training_set_path):
            raw_train = load_from_disk(f"{potential_training_set_path}")
            raw_test = loaded_dataset["test"]
            print(raw_train, "In loading mode")
            print(raw_test)
            print(f"{potential_training_set_path}/augmented_indices.json")
            with open(f"{potential_training_set_path}/augmented_indices.json", "r") as f:
                augmented_indices = {int(k): v for k,v in json.load(f).items()}
        else:
            raw_train, augmented_indices = create_augmented_dataset(
                loaded_dataset["train"], augmentation_method, n=num_augmentations, saving_path=potential_training_set_path 
            )
            raw_test = loaded_dataset["test"]

    else:
        raw_test = loaded_dataset["test"]
        raw_train = loaded_dataset["train"]
    
    return raw_test, raw_train, augmented_indices


def run_script(
    augmentation_method: AugmentationMethods | None = None, repetitions: int = 5, query_strategies: list = [], dataset: str = Datasets.IMDB.value
):
    start_time = datetime.now()
    print(f"Starting run at {start_time}. \n")
    path = (
        Path(__file__).parent / "results" / str(augmentation_method).replace(" ", "_") / dataset
    ).resolve()

    try:
        os.mkdir(path)
        print(f"Directory {path} successfully created.")
    except OSError:
        print(f"{path} already exists.")

    raw_test, raw_train, augmented_indices = create_raw_set(
        dataset, augmentation_method
    )
    print("DATASET was loaded")
    if not query_strategies:
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
        print(query_strategy)
        final_results = {}

        for rep in range(repetitions):
            saving_name = f"{query_strategy}_{num_queries}_queries_num_samples_{num_samples}_num_augmentations_{num_augmentations}.json"
            print(saving_name)
            try:
                with open((path / saving_name).resolve(), "r") as f:
                    try:
                        loaded_json = json.load(f).items()
                        final_results = {int(k): v for k, v in loaded_json}
                        keys = [int(k) for k in final_results.keys()]
                        keys.sort()
                        actual_repetition = keys[-1] + 1
                    except json.decoder.JSONDecodeError:
                        actual_repetition = rep
            except FileNotFoundError:
                actual_repetition = rep
            print("STARTING TO RUN ACTIVE LEARNING LOOP")
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

            final_results[actual_repetition] = {"0": results[0],
                                                "1": results[1] }

            with open((path / saving_name).resolve(), "w") as f:
                f.write(pd.DataFrame(final_results).to_json())

    end_time = datetime.now()
    print(f"Finished run at {end_time}. \n")
    print(f"Run took {end_time - start_time}. \n")
