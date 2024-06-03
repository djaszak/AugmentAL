import pandas as pd
from constants import (
    AUGMENTATION_METHOD_VERBOSE,
    LATEX_TABLES_VARIANCE_PATH,
    QUERY_STRATEGIES_VERBOSE,
    AugmentedPaths,
    AugmentedStrategies,
    BaselineStrategies,
    BasePaths
)
from utils import get_query_strategy_frame

import pandas as pd


def get_average_variance(folder_name: str, query_strategy: str):
    frame = get_query_strategy_frame(
        folder_name, query_strategy
    )
    variances = []
    if frame.empty:
        return 0
    for x in range(50):
        variances.append(
            frame.loc[
                frame["iterations"] == x,
                "test_accuracies",
            ].var()
        )
    var_series = pd.Series(variances)
    return var_series.mean()

def create_latex_tables_for_augmented_strategies():
    results_dict = {
            "Dataset": [],
            "Query Strategy": [],
            "Augmentation Method": [],
            "Variance": [],
            "$\Delta$ Rnd": [],
            "$\Delta$ Brk": [],
        }
    dataset_mapping = {
        "imdb": "IMDB",
        "ag_news": "AG News",
        "tweet_eval": "Tweet Eval Hate",
    }
    for folder_path in AugmentedPaths:
        # Now we would be in folder_path = AugmentedPaths.BACKTRANSLATION_TWEET
        # Which means we would be in the folder "Backtranslation/tweet_eval"
        # So we have the DataAugmentation Method and the dataset.
        # The baseline folder is the same as the folder, but with "None" instead of the augmentation method.
        # Because the latter part of the path represents the dataset.
        baseline_folder = f"None/{folder_path.value.split('/')[-1]}"
        for strategy in AugmentedStrategies:
                # We then calculate the average accuracy for each of the query strategies.
                random_baseline = get_average_variance(
                    baseline_folder, BaselineStrategies.RANDOM_SAMPLING.value
                )
                breaking_ties_baseline = get_average_variance(
                    baseline_folder, BaselineStrategies.BREAKING_TIES.value
                )
                relevant_variance = get_average_variance(
                    folder_path.value, strategy.value
                )

                results_dict["Dataset"].append(
                    dataset_mapping[folder_path.value.split("/")[-1]]
                )
                results_dict["Query Strategy"].append(
                    QUERY_STRATEGIES_VERBOSE[strategy]
                )
                results_dict["Variance"].append(relevant_variance)
                results_dict["Augmentation Method"].append(
                    AUGMENTATION_METHOD_VERBOSE[folder_path.value.split("/")[0]]
                )
                results_dict["$\Delta$ Rnd"].append(
                    random_baseline - relevant_variance
                )
                results_dict["$\Delta$ Brk"].append(
                    breaking_ties_baseline - relevant_variance
                )

    # results_dict represents all of the data that we could need for the table.
    # Now we need to split it correctly and save it to the correct files.
    frame = pd.DataFrame(results_dict)
    augmentation_method_separated_frames = [
        y for _, y in frame.groupby("Augmentation Method")
    ]
    for aug_frame in augmentation_method_separated_frames:
        query_strategy_separated_frames = [
            y for _, y in aug_frame.groupby("Query Strategy")
        ]
        for query_frame in query_strategy_separated_frames:
            current_augmentation_method = query_frame.at[
                query_frame.index[0], "Augmentation Method"
            ]
            current_query_strategy = query_frame.at[
                query_frame.index[0], "Query Strategy"
            ]
            query_frame.drop(
                columns=["Augmentation Method", "Query Strategy"], inplace=True
            )
            dataset_seperated_frames = [y for _, y in query_frame.groupby("Dataset")]
            for dataset_frame in dataset_seperated_frames:
                dataset_name = dataset_frame.at[dataset_frame.index[0], "Dataset"]
                dataset_frame.drop(columns=["Dataset"], inplace=True)
                dataset_frame.to_latex(
                    f"{LATEX_TABLES_VARIANCE_PATH}/{current_augmentation_method}_{current_query_strategy}_{dataset_name}.tex",
                    index=False,
                    float_format="{:.5f}".format,
                )

def create_latex_tables_for_base_strategies():
    results_dict = {
        "Dataset": [],
        "Query Strategy": [],
        "Variance": [],
    }
    dataset_mapping = {
        "imdb": "IMDB",
        "ag_news": "AG News",
        "tweet_eval": "Tweet Eval Hate",
    }
    for folder_path in BasePaths:
        # Now we would be in folder_path = AugmentedPaths.BACKTRANSLATION_TWEET
        # Which means we would be in the folder "Backtranslation/tweet_eval"
        # So we have the DataAugmentation Method and the dataset.
        # The baseline folder is the same as the folder, but with "None" instead of the augmentation method.
        # Because the latter part of the path represents the dataset.
        for strategy in BaselineStrategies:
                # We then calculate the average accuracy for each of the query strategies.
                relevant_variance = get_average_variance(
                    folder_path.value, strategy.value
                )

                results_dict["Dataset"].append(
                    dataset_mapping[folder_path.value.split("/")[-1]]
                )
                results_dict["Query Strategy"].append(
                    QUERY_STRATEGIES_VERBOSE[strategy]
                )
                results_dict["Variance"].append(relevant_variance)

    # results_dict represents all of the data that we could need for the table.
    # Now we need to split it correctly and save it to the correct files.
    frame = pd.DataFrame(results_dict)
    query_strategy_separated_frames = [
        y for _, y in frame.groupby("Query Strategy")
    ]
    for query_frame in query_strategy_separated_frames:
        current_query_strategy = query_frame.at[
            query_frame.index[0], "Query Strategy"
        ]
        query_frame.drop(
            columns=["Query Strategy"], inplace=True
        )
        dataset_seperated_frames = [y for _, y in query_frame.groupby("Dataset")]
        for dataset_frame in dataset_seperated_frames:
            dataset_name = dataset_frame.at[dataset_frame.index[0], "Dataset"]
            dataset_frame.drop(columns=["Dataset"], inplace=True)
            dataset_frame.to_latex(
                f"{LATEX_TABLES_VARIANCE_PATH}/base_{current_query_strategy}_{dataset_name}.tex",
                index=False,
                float_format="{:.5f}".format,
            ) 

create_latex_tables_for_augmented_strategies()
create_latex_tables_for_base_strategies()
