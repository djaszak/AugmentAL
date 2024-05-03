import enum
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class AugmentationType(enum.Enum):
    BACK_TRANSLATION_AUG = "Name:BackTranslationAug,_Action:substitute,_Method:word"
    SYNONYM_AUG = "Name:Synonym_Aug,_Aug_Src:wordnet,_Action:substitute,_Method:word"
    CONTEXTUAL_WORD_EMBS_FOR_SENTENCE = (
        "Name:ContextualWordEmbsForSentence_Aug,_Action:insert,_Method:sentence"
    )
    CONTEXTUAL_WORD_EMBS = "Name:ContextualWordEmbs_Aug,_Action:substitute,_Method:word"


class QueryStrategy(enum.Enum):
    RANDOM_SAMPLING = "RandomSampling"
    BREAKING_TIES = "BreakingTies"


def get_json_files(augmentation_type: AugmentationType):
    """
    Get all JSON files for the given query strategy and augmentation type. The files are located in the results folder.

    Args:
        - query_strategy (QueryStrategy): Query strategy.
        - augmentation_type (AugmentationType): Augmentation type.

    Returns:
        - list: List of JSON files.
    """
    root_folder = str(Path(__file__).parent / "../results")
    folder_path = os.path.join(root_folder, augmentation_type.value)
    json_files = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".json"):
            json_files.append(file_path)
    return json_files


# print(get_json_files(AugmentationType.BACK_TRANSLATION_AUG, QueryStrategy.RANDOM_SAMPLING))

# def annotate_with_stopping_line(data, **kws):
#     stopping_history = data["stopping_history"]
#     ax = plt.gca()
#     if stopping_history.any():
#         first_true_iteration = data.loc[stopping_history].iloc[0]["iterations"]
#         ax.axvline(x=first_true_iteration, color="r", linestyle="--", label="First True Stop")

QUERY_STRATEGY_COLUMN = "query_strategy"


def create_graph_for_augmentation_type(augmentation_type: AugmentationType):
    frames = []
    for file in get_json_files(augmentation_type):
        with open(file, "r") as f:
            inter_list = []
            for _, series in pd.read_json(f).items():
                frame = pd.DataFrame(series[0])
                frame[QUERY_STRATEGY_COLUMN] = os.path.basename(file).split("_")[0]
                inter_list.append(frame)
            frames.extend(inter_list)

    frame = pd.concat(frames)

    g = sns.relplot(
        data=frame,
        x="iterations",
        y="test_accuracies",
        col=QUERY_STRATEGY_COLUMN,
        kind="line",
    )
    g.figure.subplots_adjust(top=0.9)  # adjust the Figure in rp
    g.figure.suptitle(augmentation_type.value)
    # g.map_dataframe(annotate_with_stopping_line)
    for ax in g.axes.flat:
        # Get the augmentation method associated with this subplot
        augmentation_method = ax.get_title().split("=")[1].strip()

        # Filter the data for the current augmentation method
        subset_data = frame[frame[QUERY_STRATEGY_COLUMN] == augmentation_method]

        if subset_data["stopping_history"].any():
            # Find the first iteration where stop_history is True
            first_true_iteration = subset_data.loc[
                subset_data["stopping_history"]
            ].iloc[0]["iterations"]

            # Add a vertical line at the first true iteration
            ax.axvline(
                x=first_true_iteration,
                color="r",
                linestyle="--",
                label="First True Stop",
            )

            # Add a text label indicating the first true iteration
            ax.text(
                first_true_iteration,
                0.5,
                f"Iteration {first_true_iteration}",
                color="r",
                ha="right",
                va="center",
                rotation=90,
                transform=ax.get_xaxis_transform(),
            )

    plt.tight_layout()
    plt.show()


create_graph_for_augmentation_type(AugmentationType.BACK_TRANSLATION_AUG)
