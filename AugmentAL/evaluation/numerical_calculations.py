import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from constants import (
    AUGMENTATION_METHOD_VERBOSE,
    QUERY_STRATEGIES_VERBOSE,
    LATEX_IMAGES_PATH,
    AugmentedPaths,
    AugmentedStrategies,
    BaselineStrategies,
    BasePaths,
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


def get_data_frame():
    results_dict = {
        "Dataset": [],
        "Query Strategy": [],
        "DA": [],
        "i": [],
        "Acc": [],
        # "$\Delta$ Rnd": [],
        # "$\Delta$ Brk": [],
    }
    dataset_mapping = {
        "imdb": "IMDB",
        "ag_news": "AG News",
        "tweet_eval": "Tweet Eval Hate",
    }
    for folder_path in AugmentedPaths:
        for strategy in AugmentedStrategies:
            for i in [5, 10, 20, 30, 40, 50, -5, -10]:
                relevant_average_accuracy = calculate_average_over_iterations(
                    folder_path.value, i, strategy.value
                )

                results_dict["Dataset"].append(
                    dataset_mapping[folder_path.value.split("/")[-1]]
                )
                results_dict["Query Strategy"].append(
                    QUERY_STRATEGIES_VERBOSE[strategy.value]
                )
                results_dict["i"].append(i)
                results_dict["Acc"].append(relevant_average_accuracy)
                results_dict["DA"].append(
                    AUGMENTATION_METHOD_VERBOSE[folder_path.value.split("/")[0]]
                )
    for folder_path in BasePaths:
        for strategy in BaselineStrategies:
            for augmentation_method in [
                "BERT",
                "Random Swap",
                "Synonym",
                "Backtranslation",
            ]:
                for i in [5, 10, 20, 30, 40, 50, -5, -10]:
                    relevant_average_accuracy = calculate_average_over_iterations(
                        folder_path.value, i, strategy.value
                    )

                    results_dict["Dataset"].append(
                        dataset_mapping[folder_path.value.split("/")[-1]]
                    )
                    results_dict["Query Strategy"].append(
                        QUERY_STRATEGIES_VERBOSE[strategy.value]
                    )
                    results_dict["i"].append(i)
                    results_dict["Acc"].append(relevant_average_accuracy)
                    results_dict["DA"].append(augmentation_method)

    # results_dict represents all of the data that we could need for the table.
    # Now we need to split it correctly and save it to the correct files.
    return pd.DataFrame(results_dict).sort_values(
        by=["DA", "Query Strategy", "i", "Dataset"]
    )


def draw_bar_chart():
    display = pd.options.display
    display.max_columns = 1000
    display.max_rows = 10_000
    display.max_colwidth = 199
    display.width = 1000
    frame = get_data_frame()
    g = sns.catplot(
        frame,
        kind="bar",
        x="i",
        y="Acc",
        col="DA",
        row="Dataset",
        height=4,
        aspect=0.5,
        hue="Query Strategy",
    )

    # Adjusting subplot parameters to reduce overlap
    g.figure.subplots_adjust(wspace=0.1, hspace=0.2)
    g.set_axis_labels("Iterations", "Test Accuracy")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")

    plt.savefig(f"{LATEX_IMAGES_PATH}/average_accuracy_over_iterations.png")
    # plt.show()


draw_bar_chart()
