import pandas as pd
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
import seaborn as sns
import matplotlib.pyplot as plt


def get_average_variance(folder_name: str, query_strategy: str):
    frame = get_query_strategy_frame(folder_name, query_strategy)
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


def create_frame():
    results_dict = {
        "Dataset": [],
        "Query Strategy": [],
        "DA": [],
        "Variance": [],
    }
    dataset_mapping = {
        "imdb": "IMDB",
        "ag_news": "AG News",
        "tweet_eval": "Tweet Eval Hate",
    }
    for folder_path in AugmentedPaths:
        for strategy in AugmentedStrategies:
            relevant_variance = get_average_variance(folder_path.value, strategy.value)

            results_dict["Dataset"].append(
                dataset_mapping[folder_path.value.split("/")[-1]]
            )
            results_dict["Query Strategy"].append(
                QUERY_STRATEGIES_VERBOSE[strategy.value]
            )
            results_dict["Variance"].append(relevant_variance)
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
                relevant_variance = get_average_variance(
                    folder_path.value, strategy.value
                )

                results_dict["Dataset"].append(
                    dataset_mapping[folder_path.value.split("/")[-1]]
                )
                results_dict["Query Strategy"].append(
                    QUERY_STRATEGIES_VERBOSE[strategy.value]
                )
                results_dict["Variance"].append(relevant_variance)
                results_dict["DA"].append(augmentation_method)
    return pd.DataFrame(results_dict)


def draw_bar_chart():
    display = pd.options.display
    display.max_columns = 1000
    display.max_rows = 10_000
    display.max_colwidth = 199
    display.width = 1000
    frame = create_frame()
    frame.replace(
        {
            "Extended Search Space": "ESS",
            "Random Sampling": "Rnd",
            "Breaking Ties": "Brk",
            "Average Across Augmented": "AAA",
            "Extended Outcome": "EO",
        },
        inplace=True,
    )
    g = sns.catplot(
        frame,
        kind="bar",
        x="Query Strategy",
        y="Variance",
        col="DA",
        row="Dataset",
        height=4,
        aspect=0.5,
        hue="Query Strategy",
        # facet_kws={
        #     "margin_titles": True,
        # },
    )

    # Adjusting subplot parameters to reduce overlap
    g.figure.subplots_adjust(wspace=0.1, hspace=0.2)
    g.set_axis_labels("Query Strategy", "Variance")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")


    plt.savefig(f"{LATEX_IMAGES_PATH}/variance.png")
    print(f"Saved variance plot to {LATEX_IMAGES_PATH}/variance.png")
    # plt.show()

draw_bar_chart()
