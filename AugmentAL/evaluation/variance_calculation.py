import pandas as pd
from constants import (
    AUGMENTATION_METHOD_VERBOSE,
    LATEX_TABLES_VARIANCE_PATH,
    QUERY_STRATEGIES_VERBOSE,
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
    frame = get_data_frame()
    baseline_data = frame[frame["DA"] == "None"]
    plot_data = frame[frame["DA"] != "None"]
    # print(plot_data)
    # qs_plots = plot_data.groupby(["Query Strategy"])
    # for i, (qs, qs_frame) in enumerate(qs_plots):
    g = sns.catplot(
        plot_data,
        kind="bar",
        x="i",
        y="Acc",
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
    g.set_axis_labels("Iterations", "Test Accuracy")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")

    plt.savefig(f"{LATEX_IMAGES_PATH}/average_accuracy_over_iterations.png")
    plt.show()


def create_latex_tables_for_augmented_strategies():
    # results_dict represents all of the data that we could need for the table.
    # Now we need to split it correctly and save it to the correct files.
    frame = create_frame()
    # frame.replace(
    #     {
    #         "Backtranslation": "Back",
    #         "Random Swap": "Swap",
    #         "Synonym": "Syn",
    #     },
    #     inplace=True,
    # )
    # augmentation_method_separated_frames = [
    #     y for _, y in frame.groupby("DA")
    # ]
    # for aug_frame in augmentation_method_separated_frames:
    query_strategy_separated_frames = [y for _, y in frame.groupby("Query Strategy")]
    for query_frame in query_strategy_separated_frames:
        current_augmentation_method = query_frame.at[query_frame.index[0], "DA"]
        current_query_strategy = query_frame.at[query_frame.index[0], "Query Strategy"]
        query_frame.drop(columns=["Query Strategy"], inplace=True)
        dataset_seperated_frames = [y for _, y in query_frame.groupby("Dataset")]
        for j, dataset_frame in enumerate(dataset_seperated_frames):
            dataset_name = dataset_frame.at[dataset_frame.index[0], "Dataset"]
            dataset_frame.drop(columns=["Dataset"], inplace=True)
            if j > 0:
                dataset_frame.drop(columns=["DA"], inplace=True)
            dataset_frame.to_latex(
                f"{LATEX_TABLES_VARIANCE_PATH}/{current_query_strategy}_{dataset_name}.tex",
                index=False,
                float_format="{:.5f}".format,
            )


def draw_bar_chart():
    display = pd.options.display
    display.max_columns = 1000
    display.max_rows = 10_000
    display.max_colwidth = 199
    display.width = 1000
    frame = get_data_frame()
    baseline_data = frame[frame["DA"] == "None"]
    plot_data = frame[frame["DA"] != "None"]
    # print(plot_data)
    # qs_plots = plot_data.groupby(["Query Strategy"])
    # for i, (qs, qs_frame) in enumerate(qs_plots):
    g = sns.catplot(
        plot_data,
        kind="bar",
        x="i",
        y="Acc",
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
    g.set_axis_labels("Iterations", "Test Accuracy")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")

    plt.savefig(f"{LATEX_IMAGES_PATH}/average_accuracy_over_iterations.png")
    plt.show


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
        # So we have the DataDA and the dataset.
        # The baseline folder is the same as the folder, but with "None" instead of the DA.
        # Because the latter part of the path represents the dataset.
        for strategy in BaselineStrategies:
            # We then calculate the average accuracy for each of the query strategies.
            relevant_variance = get_average_variance(folder_path.value, strategy.value)

            results_dict["Dataset"].append(
                dataset_mapping[folder_path.value.split("/")[-1]]
            )
            results_dict["Query Strategy"].append(
                QUERY_STRATEGIES_VERBOSE[strategy.value]
            )
            results_dict["Variance"].append(relevant_variance)

    # results_dict represents all of the data that we could need for the table.
    # Now we need to split it correctly and save it to the correct files.
    frame = pd.DataFrame(results_dict)
    query_strategy_separated_frames = [y for _, y in frame.groupby("Query Strategy")]
    for query_frame in query_strategy_separated_frames:
        current_query_strategy = query_frame.at[query_frame.index[0], "Query Strategy"]
        query_frame.drop(columns=["Query Strategy"], inplace=True)
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
# create_latex_tables_for_base_strategies()
