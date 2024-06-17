import pandas as pd
from constants import (
    AUGMENTATION_METHOD_VERBOSE,
    LATEX_TABLES_STOP_PATH,
    QUERY_STRATEGIES_VERBOSE,
    AugmentedPaths,
    AugmentedStrategies,
    BaselineStrategies,
    BasePaths,
    STOPPING_CRITERIA,
    STOPPING_CRITERIA_VERBOSE_SHORT,
)
from utils import get_query_strategy_frame


def calculate_iter_and_acc(folder_name, query_strategy, criterion):
    query_strategy_filtered_frame = get_query_strategy_frame(
        folder_name, query_strategy
    )
    if not query_strategy_filtered_frame.empty:
        if query_strategy_filtered_frame[criterion].any():
            # Calculate the average iteration
            average_true_iteration = int(
                query_strategy_filtered_frame.loc[
                    query_strategy_filtered_frame[criterion] == True, "iterations"
                ].mean()
            )

            # Calculate the average accuracy
            average_accuracy = query_strategy_filtered_frame.loc[
                query_strategy_filtered_frame[criterion] == True, "test_accuracies"
            ].mean()

            # print(average_true_iteration, average_accuracy)
            return average_true_iteration, average_accuracy
        return 0, 0
    return 0, 0


def create_latex_tables_for_augmented_strategies():
    results_dict = {
        "Dataset": [],
        "Query Strategy": [],
        "Stopping Criterion": [],
        "DA": [],
        "Iter/Acc": [],
        "Iter/Acc Rnd": [],
        "Iter/Acc Brk": [],
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
            for criterion in STOPPING_CRITERIA:
                average_true_iteration, average_accuracy = calculate_iter_and_acc(
                    folder_path.value, strategy.value, criterion
                )
                breaking_true_iteration, breaking_accuracy = calculate_iter_and_acc(
                    baseline_folder, BaselineStrategies.BREAKING_TIES.value, criterion
                )
                random_true_iteration, random_accuracy = calculate_iter_and_acc(
                    baseline_folder, BaselineStrategies.BREAKING_TIES.value, criterion
                )
                # Update the results dictionary
                results_dict["Query Strategy"].append(
                    QUERY_STRATEGIES_VERBOSE[strategy.value]
                )
                results_dict["Stopping Criterion"].append(
                    STOPPING_CRITERIA_VERBOSE_SHORT[criterion]
                )
                results_dict["DA"].append(
                    AUGMENTATION_METHOD_VERBOSE[folder_path.value.split("/")[0]]
                )
                results_dict["Iter/Acc"].append(
                    f"{average_true_iteration}/{average_accuracy}"
                )
                results_dict["Iter/Acc Brk"].append(
                    f"{breaking_true_iteration}/{breaking_accuracy}"
                )
                results_dict["Iter/Acc Rnd"].append(
                    f"{random_true_iteration}/{random_accuracy}"
                )
                results_dict["Dataset"].append(
                    dataset_mapping[folder_path.value.split("/")[-1]]
                )
                # We then calculate the average accuracy for each of the query strategies.

    # Now we need to split it correctly and save it to the correct files.
    frame = pd.DataFrame(results_dict).sort_values(
        by=["DA", "Query Strategy", "Dataset"]
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
                f"{LATEX_TABLES_STOP_PATH}/{current_query_strategy}_{dataset_name}.tex",
                index=False,
                float_format="{:.2f}".format,
            )

    return results_dict


results = create_latex_tables_for_augmented_strategies()

# for key, Value in results.items():
#     print(f"{key} : {Value}")

print(pd.DataFrame(results))

# def create_latex_tables_for_augmented_strategies():
#     results_dict = {
#         "Dataset": [],
#         "Query Strategy": [],
#         "Stopping Criterion": [],
#         "DA": [],
#         "Iter/Acc": [],
#         "$\Delta$ Rnd": [],
#         "$\Delta$ Brk": [],
#     }
#     dataset_mapping = {
#         "imdb": "IMDB",
#         "ag_news": "AG News",
#         "tweet_eval": "Tweet Eval Hate",
#     }
#     for folder_path in AugmentedPaths:
#         # Now we would be in folder_path = AugmentedPaths.BACKTRANSLATION_TWEET
#         # Which means we would be in the folder "Backtranslation/tweet_eval"
#         # So we have the DataAugmentation Method and the dataset.
#         # The baseline folder is the same as the folder, but with "None" instead of the DA.
#         # Because the latter part of the path represents the dataset.
#         baseline_folder = f"None/{folder_path.value.split('/')[-1]}"
#         for strategy in AugmentedStrategies:
#             for criterion in STOPPING_CRITERIA:
#                 if subset_data[criterion].any():
#                     # Calculate the average iteration
#                     average_true_iteration = int(
#                         subset_data.loc[subset_data[criterion] == True, ITERATION_COLUMN].mean()
#                     )

#                     # Calculate the average accuracy
#                     average_accuracy = subset_data.loc[
#                         subset_data[criterion] == True, ACCURACY_COLUMN
#                     ].mean()

#                     # Update the results dictionary
#                     results_dict["Stopping Criterion"].append(STOPPING_CRITERIA_VERBOSE_SHORT[criterion])
#                     results_dict["Augmentation Method"].append(AUGMENTATION_METHOD_VERBOSE[augmentation_method])
#                     results_dict["Average Iteration"].append(average_true_iteration)
#                     results_dict["Average Accuracy"].append(average_accuracy)
#                     results_dict["Dataset"].append(DATASETS_VERBOSE[dataset])
#                 # We then calculate the average accuracy for each of the query strategies.
#             random_baseline = calculate_average_over_iterations(
#                 baseline_folder, i, BaselineStrategies.RANDOM_SAMPLING.value
#             )
#             breaking_ties_baseline = calculate_average_over_iterations(
#                 baseline_folder, i, BaselineStrategies.BREAKING_TIES.value
#             )
#             relevant_average_accuracy = calculate_average_over_iterations(
#                 folder_path.value, i, strategy.value
#             )

#             results_dict["Dataset"].append(
#                 dataset_mapping[folder_path.value.split("/")[-1]]
#             )
#             results_dict["Query Strategy"].append(
#                 QUERY_STRATEGIES_VERBOSE[strategy.value]
#             )
#             results_dict["Acc"].append(relevant_average_accuracy)
#             results_dict["DA"].append(
#                 AUGMENTATION_METHOD_VERBOSE[folder_path.value.split("/")[0]]
#             )
#             results_dict["$\Delta$ Rnd"].append(
#                 relevant_average_accuracy - random_baseline
#             )
#             results_dict["$\Delta$ Brk"].append(
#                 relevant_average_accuracy - breaking_ties_baseline
#             )

#     # results_dict represents all of the data that we could need for the table.
#     # Now we need to split it correctly and save it to the correct files.
#     frame = pd.DataFrame(results_dict).sort_values(
#         by=["DA", "Query Strategy", "i", "Dataset"]
#     )
#     frame.replace(
#         {
#             "Backtranslation": "Back",
#             "Random Swap": "Swap",
#             "Synonym": "Syn",
#         },
#         inplace=True,
#     )
#     # augmentation_method_separated_frames = [
#     #     y for _, y in frame.groupby("DA")
#     # ]
#     query_strategy_separated_frames = [
#         y for _, y in frame.groupby("Query Strategy")
#     ]
#     for j, query_frame in enumerate(query_strategy_separated_frames):
#     # for aug_frame in augmentation_method_separated_frames:
#         # current_augmentation_method = query_frame.at[
#         #     query_frame.index[0], "DA"
#         # ]
#         current_query_strategy = query_frame.at[
#             query_frame.index[0], "Query Strategy"
#         ]
#         query_frame.drop(
#             columns=["Query Strategy"], inplace=True
#         )

#         dataset_seperated_frames = [y for _, y in query_frame.groupby("Dataset")]
#         for j, dataset_frame in enumerate(dataset_seperated_frames):
#             dataset_name = dataset_frame.at[dataset_frame.index[0], "Dataset"]
#             dataset_frame.drop(columns=["Dataset"], inplace=True)
#             if j > 0:
#                 dataset_frame.drop(columns=["i"], inplace=True)

#                 dataset_frame.drop(
#                     columns=["DA"], inplace=True
#                 )
#             dataset_frame.to_latex(
#                 f"{LATEX_TABLES_PATH}/{current_query_strategy}_{dataset_name}.tex",
#                 index=False,
#                 float_format="{:.2f}".format,
#             )

# def create_latex_tables_for_baseline_strategies():
#     results_dict = {
#         "Dataset": [],
#         "Query Strategy": [],
#         "i": [],
#         "Acc": [],
#     }
#     dataset_mapping = {
#         "imdb": "IMDB",
#         "ag_news": "AG News",
#         "tweet_eval": "Tweet Eval Hate",
#     }
#     for folder_path in BasePaths:
#         # Now we would be in folder_path = AugmentedPaths.BACKTRANSLATION_TWEET
#         # Which means we would be in the folder "Backtranslation/tweet_eval"
#         # So we have the DataAugmentation Method and the dataset.
#         # The baseline folder is the same as the folder, but with "None" instead of the DA.
#         # Because the latter part of the path represents the dataset.
#         for strategy in BaselineStrategies:
#             for i in [5, 10, 20, 30, 40, 50, -5, -10]:
#                 # We then calculate the average accuracy for each of the query strategies.
#                 relevant_average_accuracy = calculate_average_over_iterations(
#                     folder_path.value, i, strategy.value
#                 )

#                 results_dict["Dataset"].append(
#                     dataset_mapping[folder_path.value.split("/")[-1]]
#                 )
#                 results_dict["Query Strategy"].append(
#                     QUERY_STRATEGIES_VERBOSE[strategy]
#                 )
#                 results_dict["i"].append(i)
#                 results_dict["Acc"].append(relevant_average_accuracy)

#     # results_dict represents all of the data that we could need for the table.
#     # Now we need to split it correctly and save it to the correct files.
#     frame = pd.DataFrame(results_dict).sort_values(
#         by=["Query Strategy", "i", "Dataset"]
#     )
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

# create_latex_tables_for_augmented_strategies()
# # create_latex_tables_for_baseline_strategies()
