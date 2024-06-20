import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from constants import (
    AUGMENTATION_METHOD_VERBOSE,
    LATEX_IMAGES_PATH,
    QUERY_STRATEGIES_VERBOSE,
    STOPPING_CRITERIA,
    STOPPING_CRITERIA_VERBOSE_SHORT,
    AugmentedPaths,
    AugmentedStrategies,
    BaselineStrategies,
    BasePaths,
)
from utils import get_query_strategy_frame


def calculate_iter_and_acc(folder_name, query_strategy, criterion):
    query_strategy_filtered_frame = get_query_strategy_frame(
        folder_name, query_strategy
    )
    if not query_strategy_filtered_frame.empty:
        if query_strategy_filtered_frame[criterion].any():
            average_true_iteration = int(
                query_strategy_filtered_frame.loc[
                    query_strategy_filtered_frame[criterion] == True, "iterations"
                ].mean()
            )
            average_accuracy = query_strategy_filtered_frame.loc[
                query_strategy_filtered_frame[criterion] == True, "test_accuracies"
            ].mean()
            return average_true_iteration, average_accuracy
        return 0, 0
    return 0, 0


def plot_stopping_criteria(
    AugmentedPaths,
    AugmentedStrategies,
    STOPPING_CRITERIA,
    calculate_iter_and_acc,
    BaselineStrategies,
    QUERY_STRATEGIES_VERBOSE,
    STOPPING_CRITERIA_VERBOSE_SHORT,
    AUGMENTATION_METHOD_VERBOSE,
):
    results_dict = {
        "Dataset": [],
        "Query Strategy": [],
        "Stopping Criterion": [],
        "DA": [],
        "Average Iteration": [],
        "Average Accuracy": [],
    }
    dataset_mapping = {
        "imdb": "IMDB",
        "ag_news": "AG News",
        "tweet_eval": "Tweet",
    }

    for folder_path in AugmentedPaths:
        for strategy in AugmentedStrategies:
            for criterion in STOPPING_CRITERIA:
                iter_and_acc = calculate_iter_and_acc(
                    folder_path.value, strategy.value, criterion
                )
                results_dict["Query Strategy"].append(
                    QUERY_STRATEGIES_VERBOSE[strategy.value]
                )
                results_dict["Stopping Criterion"].append(
                    STOPPING_CRITERIA_VERBOSE_SHORT[criterion]
                )
                results_dict["DA"].append(
                    AUGMENTATION_METHOD_VERBOSE[folder_path.value.split("/")[0]]
                )
                results_dict["Average Iteration"].append(iter_and_acc[0])
                results_dict["Average Accuracy"].append(iter_and_acc[1])
                results_dict["Dataset"].append(
                    dataset_mapping[folder_path.value.split("/")[-1]]
                )

    for folder_path in BasePaths:
        for strategy in BaselineStrategies:
            for criterion in STOPPING_CRITERIA:
                for augmentation_method in [
                    "BERT",
                    "Random Swap",
                    "Synonym",
                    "Backtranslation",
                ]:
                    iter_and_acc = calculate_iter_and_acc(
                        folder_path.value, strategy.value, criterion
                    )
                    if iter_and_acc == (0, 0):
                        continue
                    else:
                        average_true_iteration, average_accuracy = iter_and_acc
                    results_dict["Query Strategy"].append(
                        QUERY_STRATEGIES_VERBOSE[strategy.value]
                    )
                    results_dict["Stopping Criterion"].append(
                        STOPPING_CRITERIA_VERBOSE_SHORT[criterion]
                    )
                    results_dict["DA"].append(augmentation_method)
                    results_dict["Average Iteration"].append(average_true_iteration)
                    results_dict["Average Accuracy"].append(average_accuracy)
                    results_dict["Dataset"].append(
                        dataset_mapping[folder_path.value.split("/")[-1]]
                    )

    # Convert the results dictionary to a DataFrame
    df = pd.DataFrame(results_dict)

    # Plot for each stopping criterion
    for criterion in STOPPING_CRITERIA:
        criterion_data = df[
            df["Stopping Criterion"] == STOPPING_CRITERIA_VERBOSE_SHORT[criterion]
        ]

        # Pivot for heatmap
        heatmap_data = criterion_data.pivot_table(
            index=["Dataset", "DA"],
            columns="Query Strategy",
            values="Average Iteration",
        )
        criterion_data_without_iteration = criterion_data.drop(
            columns=["Average Iteration"]
        )
        criterion_data_without_iteration.rename(
            columns={"Average Accuracy": "Average Iteration"}, inplace=True
        )
        heatmap_data_for_accuracy = criterion_data_without_iteration.pivot_table(
            index=["Dataset", "DA"],
            columns="Query Strategy",
            values="Average Iteration",
        )

        # Plotting the heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            heatmap_data,
            annot=True,
            cmap="YlGnBu",
            cbar_kws={"label": "Average Iteration"},
            annot_kws={"va": "bottom"},
        )
        sns.heatmap(
            heatmap_data_for_accuracy,
            annot=True,
            fmt=".2f",  # format to show 2 decimal places
            cmap="YlGnBu",
            cbar=False,
            annot_kws={"va": "top"},
        )
        plt.title(f"Average Iteration for {STOPPING_CRITERIA_VERBOSE_SHORT[criterion]}")
        plt.xlabel("Query Strategy")
        plt.ylabel("Dataset and Data Augmentation")
        plt.tight_layout()

        plot_filename = f"{LATEX_IMAGES_PATH}/stopping_criteria_heatmap_{STOPPING_CRITERIA_VERBOSE_SHORT[criterion]}.png"
        plt.savefig(plot_filename, dpi=300)
        # plt.show()


# Call the function with the appropriate parameters
plot_stopping_criteria(
    AugmentedPaths,
    AugmentedStrategies,
    STOPPING_CRITERIA,
    calculate_iter_and_acc,
    BaselineStrategies,
    QUERY_STRATEGIES_VERBOSE,
    STOPPING_CRITERIA_VERBOSE_SHORT,
    AUGMENTATION_METHOD_VERBOSE,
)
