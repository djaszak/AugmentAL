import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from constants import (
    AUGMENTATION_METHOD_VERBOSE,
    LATEX_TABLES_PATH,
    QUERY_STRATEGIES_VERBOSE,
    LATEX_IMAGES_PATH,
    AugmentedPaths,
    AugmentedStrategies,
    BaselineStrategies,
    BasePaths,
    AugmentationType,
)
from utils import get_query_strategy_frame, create_complete_frame_for_all_folders


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
        "DA": [],
        "i": [],
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
        # The baseline folder is the same as the folder, but with "None" instead of the DA.
        # Because the latter part of the path represents the dataset.
        baseline_folder = f"None/{folder_path.value.split('/')[-1]}"
        for strategy in AugmentedStrategies:
            for i in [5, 10, 20, 30, 40, 50, -5, -10]:
                # We then calculate the average accuracy for each of the query strategies.
                random_baseline = calculate_average_over_iterations(
                    baseline_folder, i, BaselineStrategies.RANDOM_SAMPLING.value
                )
                breaking_ties_baseline = calculate_average_over_iterations(
                    baseline_folder, i, BaselineStrategies.BREAKING_TIES.value
                )
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
                results_dict["$\Delta$ Rnd"].append(
                    relevant_average_accuracy - random_baseline
                )
                results_dict["$\Delta$ Brk"].append(
                    relevant_average_accuracy - breaking_ties_baseline
                )

    # results_dict represents all of the data that we could need for the table.
    # Now we need to split it correctly and save it to the correct files.
    frame = pd.DataFrame(results_dict).sort_values(
        by=["DA", "Query Strategy", "i", "Dataset"]
    )
    frame.replace(
        {
            "Backtranslation": "Back",
            "Random Swap": "Swap",
            "Synonym": "Syn",
        },
        inplace=True,
    )
    # augmentation_method_separated_frames = [
    #     y for _, y in frame.groupby("DA")
    # ]
    query_strategy_separated_frames = [y for _, y in frame.groupby("Query Strategy")]
    for j, query_frame in enumerate(query_strategy_separated_frames):
        # for aug_frame in augmentation_method_separated_frames:
        # current_augmentation_method = query_frame.at[
        #     query_frame.index[0], "DA"
        # ]
        current_query_strategy = query_frame.at[query_frame.index[0], "Query Strategy"]
        query_frame.drop(columns=["Query Strategy"], inplace=True)

        dataset_seperated_frames = [y for _, y in query_frame.groupby("Dataset")]
        for j, dataset_frame in enumerate(dataset_seperated_frames):
            dataset_name = dataset_frame.at[dataset_frame.index[0], "Dataset"]
            dataset_frame.drop(columns=["Dataset"], inplace=True)
            if j > 0:
                dataset_frame.drop(columns=["i"], inplace=True)

                dataset_frame.drop(columns=["DA"], inplace=True)
            dataset_frame.to_latex(
                f"{LATEX_TABLES_PATH}/{current_query_strategy}_{dataset_name}.tex",
                index=False,
                float_format="{:.2f}".format,
            )


def get_data_frame():
    # frame, n_frames = create_complete_frame_for_all_folders()

    # # Replacing values in the DataFrame for readability
    # frame.replace(
    #     {
    #         AugmentationType.BACK_TRANSLATION_AUG.value: "Backtranslation",
    #         AugmentationType.SYNONYM_AUG.value: "Synonym",
    #         AugmentationType.RANDOM_SWAP.value: "Random Swap",
    #         AugmentationType.CONTEXTUAL_WORD_EMBS.value: "BERT",
    #         "tweet_eval": "Tweet Eval Hate",
    #         "imdb": "IMDB",
    #         "ag_news": "AG News",
    #         BaselineStrategies.RANDOM_SAMPLING.value: "Random Sampling",
    #         BaselineStrategies.BREAKING_TIES.value: "Breaking Ties",
    #         AugmentedStrategies.AUGMENTED_OUTCOME.value: "Extended Outcome",
    #         AugmentedStrategies.AUGMENTED_SEARCH_SPACE.value: "Extended Search Space",
    #         AugmentedStrategies.AVERAGE_ACROSS_AUGMENTED.value: "AAA",
    #     },
    #     inplace=True,
    # )

    # # Renaming columns for readability
    # frame.rename(
    #     columns={
    #         "dataset": "Dataset",
    #         "augmentation_method": "Augmentation Method",
    #         "query_strategy": "Query Strategy",
    #     },
    #     inplace=True,
    # )

    # # Separate the baseline data
    # baseline_data = frame[frame["Augmentation Method"] == "None"]
    # plot_data = frame[frame["Augmentation Method"] != "None"]

    # return baseline_data, plot_data
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


draw_bar_chart()
# create_latex_tables_for_augmented_strategies()
# create_latex_tables_for_baseline_strategies()

# def create_latex_tables_for_baseline_strategies():
#     frame = get_data_frame()
#     query_strategy_separated_frames = [
#         y for _, y in frame.groupby("Query Strategy")
#     ]
#     for query_frame in query_strategy_separated_frames:
#         current_query_strategy = query_frame.at[
#             query_frame.index[0], "Query Strategy"
#         ]
#         query_frame.drop(
#             columns=["Query Strategy"], inplace=True
#         )
#         dataset_separated_frames = [y for _, y in query_frame.groupby("Dataset")]
#         for j, dataset_frame in enumerate(dataset_separated_frames):
#             dataset_name = dataset_frame.at[dataset_frame.index[0], "Dataset"]
#             if j > 0:
#                 dataset_frame.drop(columns=["i"], inplace=True)
#             dataset_frame.to_latex(
#                     f"{LATEX_TABLES_PATH}/base_{current_query_strategy}_{dataset_name}.tex",
#                     index=False,
#                     float_format="{:.2f}".format,
#                 )
