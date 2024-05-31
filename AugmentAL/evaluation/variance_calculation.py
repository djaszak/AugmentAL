from utils import get_query_strategy_frame
from constants import (
    FolderPaths,
    AugmentedStrategies,
    AUGMENTATION_METHODS_DICT,
    QUERY_STRATEGIES_VERBOSE,
    Datasets,
    DATASETS_VERBOSE,
    LATEX_TABLES_VARIANCE_PATH,
)

import pandas as pd


def create_variance_dict(frame):
    variance_dict = {
        "iterations": [],
        "variance": [],
    }

    for x in range(50):
        variance_for_given_iteration = frame.loc[
            frame["iterations"] == x,
            "test_accuracies",
        ].var()
        variance_dict["iterations"].append(x)
        variance_dict["variance"].append(variance_for_given_iteration)

    return variance_dict


def get_average_variance(frame):
    variances = []
    for x in range(50):
        variances.append(
            frame.loc[
                frame["iterations"] == x,
                "test_accuracies",
            ].var()
        )
    var_series = pd.Series(variances)
    return var_series.mean()


def get_variance_results_from_folder(folder: FolderPaths):
    average_variance_dict = {
        "Dataset": [],
        "Query Strategy": [],
        # "DA Method": [],
        "Average Variance": [],
    }
    for dataset in Datasets:
        for query_strategy in AugmentedStrategies:
            if (
                query_strategy == AugmentedStrategies.RANDOM_SAMPLING
                or query_strategy == AugmentedStrategies.BREAKING_TIES
            ):
                continue
            frame = get_query_strategy_frame(folder, query_strategy.value)
            average_variance_dict["Dataset"].append(DATASETS_VERBOSE[dataset])
            average_variance_dict["Query Strategy"].append(
                QUERY_STRATEGIES_VERBOSE[query_strategy]
            )
            average_variance_dict["Average Variance"].append(
                get_average_variance(frame)   
            )

    return average_variance_dict

for path, method_name in AUGMENTATION_METHODS_DICT.items():
    # One iteration resembles one folder
    # One folder resembles one specifically augmented dataset
    # A path looks like "{AugmentationMethod}/{Dataset}"
    frame = pd.DataFrame(
        get_variance_results_from_folder(path)
    ).sort_values(by=["Query Strategy", "Dataset"])
    frame.to_latex(
            f"{LATEX_TABLES_VARIANCE_PATH}/{method_name}.tex",
            index=False,
            float_format="{:.4f}".format,
        )

