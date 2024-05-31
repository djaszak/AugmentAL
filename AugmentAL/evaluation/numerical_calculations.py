import pandas as pd
from constants import (
    AUGMENTATION_METHOD_VERBOSE,
    LATEX_TABLES_PATH,
    QUERY_STRATEGIES_VERBOSE,
    AugmentedPaths,
    AugmentedStrategies,
    BaselineStrategies,
    BasePaths
)
from utils import get_query_strategy_frame


def calculate_average_over_iterations(
    folder_name: str, iterations: int, query_strategy: str
):
    query_strategy_filtered_frame = get_query_strategy_frame(
        folder_name, query_strategy
    )
    if not query_strategy_filtered_frame.empty:
        if iterations < 0:
            rows_lte_iterations_accuracies = query_strategy_filtered_frame.loc[
                query_strategy_filtered_frame["iterations"] >= 51 + iterations,
                "test_accuracies",
            ]
        else:
            rows_lte_iterations_accuracies = query_strategy_filtered_frame.loc[
                query_strategy_filtered_frame["iterations"] <= iterations,
                "test_accuracies",
            ]
        mean = rows_lte_iterations_accuracies.mean()
        if mean == "nan":
            print(folder_name, iterations, query_strategy)
        return mean
    return 0


def create_latex_tables_for_augmented_strategies():
    results_dict = {
        "Dataset": [],
        "Query Strategy": [],
        "Augmentation Method": [],
        "n": [],
        "Acc": [],
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
            for n in [5, 10, 20, 30, 40, 50]:
                # We then calculate the average accuracy for each of the query strategies.
                random_baseline = calculate_average_over_iterations(
                    baseline_folder, n, BaselineStrategies.RANDOM_SAMPLING.value
                )
                breaking_ties_baseline = calculate_average_over_iterations(
                    baseline_folder, n, BaselineStrategies.BREAKING_TIES.value
                )
                relevant_average_accuracy = calculate_average_over_iterations(
                    folder_path.value, n, strategy.value
                )

                results_dict["Dataset"].append(
                    dataset_mapping[folder_path.value.split("/")[-1]]
                )
                results_dict["Query Strategy"].append(
                    QUERY_STRATEGIES_VERBOSE[strategy]
                )
                results_dict["n"].append(n if n >= 0 else 50 + n)
                results_dict["Acc"].append(relevant_average_accuracy)
                results_dict["Augmentation Method"].append(
                    AUGMENTATION_METHOD_VERBOSE[folder_path.value.split("/")[0]]
                )
                results_dict["$\Delta$ Rnd"].append(
                    relevant_average_accuracy - random_baseline
                )
                results_dict["$\Delta$ Brk"].append(
                    relevant_average_accuracy - breaking_ties_baseline
                )

    # results_dict represents all of the data that we could need for the table.
    # Now we need to split it correctly and save it to the correct files.
    frame = pd.DataFrame(results_dict).sort_values(
        by=["Query Strategy", "n", "Dataset"]
    )
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
            for j, dataset_frame in enumerate(dataset_seperated_frames):
                dataset_name = dataset_frame.at[dataset_frame.index[0], "Dataset"]
                dataset_frame.drop(columns=["Dataset"], inplace=True)
                if j > 0:
                    dataset_frame.drop(columns=["n"], inplace=True)
                dataset_frame.to_latex(
                    f"{LATEX_TABLES_PATH}/{current_augmentation_method}_{current_query_strategy}_{dataset_name}.tex",
                    index=False,
                    float_format="{:.2f}".format,
                )

def create_latex_tables_for_baseline_strategies():
    results_dict = {
        "Dataset": [],
        "Query Strategy": [],
        "n": [],
        "Acc": [],
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
            for n in [5, 10, 20, 30, 40, 50]:
                # We then calculate the average accuracy for each of the query strategies.
                relevant_average_accuracy = calculate_average_over_iterations(
                    folder_path.value, n, strategy.value
                )

                results_dict["Dataset"].append(
                    dataset_mapping[folder_path.value.split("/")[-1]]
                )
                results_dict["Query Strategy"].append(
                    QUERY_STRATEGIES_VERBOSE[strategy]
                )
                results_dict["n"].append(n if n >= 0 else 50 + n)
                results_dict["Acc"].append(relevant_average_accuracy)

    # results_dict represents all of the data that we could need for the table.
    # Now we need to split it correctly and save it to the correct files.
    frame = pd.DataFrame(results_dict).sort_values(
        by=["Query Strategy", "n", "Dataset"]
    )
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
        dataset_separated_frames = [y for _, y in query_frame.groupby("Dataset")]
        for j, dataset_frame in enumerate(dataset_separated_frames):
            dataset_name = dataset_frame.at[dataset_frame.index[0], "Dataset"]
            if j > 0:
                dataset_frame.drop(columns=["n"], inplace=True)
            dataset_frame.to_latex(
                    f"{LATEX_TABLES_PATH}/base_{current_query_strategy}_{dataset_name}.tex",
                    index=False,
                    float_format="{:.2f}".format,
                )

create_latex_tables_for_baseline_strategies()