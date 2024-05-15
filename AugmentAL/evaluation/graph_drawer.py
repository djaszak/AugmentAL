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


def get_json_files(folder_name: str):
    """
    Get all JSON files for the given query strategy and augmentation type. The files are located in the results folder.

    Args:
        - query_strategy (QueryStrategy): Query strategy.
        - augmentation_type (str): The augmentation type, as it used to determine the correct folder.
            no_augmentation is stored in the folder "None", therefore str should be valid, too.

    Returns:
        - list: List of JSON files.
    """
    root_folder = str(Path(__file__).parent / "../results")
    folder_path = os.path.join(root_folder, folder_name)
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
STOPPING_CRITERIA = [
    "kappa_average_conservative_history",
    "kappa_average_middle_ground_history",
    "kappa_average_aggressive_history",
    "delta_f_score_conservative_history",
    "delta_f_score_middle_ground_history",
    "delta_f_score_aggressive_history",
    "classification_change_conservative_history",
    "classification_change_middle_ground_history",
    "classification_change_aggressive_history",
]

def pad_dict_list(dict_list, padel):
    lmax = 0
    for lname in dict_list.keys():
        lmax = max(lmax, len(dict_list[lname]))
    for lname in dict_list.keys():
        ll = len(dict_list[lname])
        if  ll < lmax:
            dict_list[lname] += [padel] * (lmax - ll)
    return dict_list

def create_graph_for_augmentation_type(folder_name: str):
    frames = []
    for file in get_json_files(folder_name):
        with open(file, "r") as f:
            inter_list = []
            for _, series in pd.read_json(f).items():
                # Because of a small oversight, the stopping criteria do have one value less
                # than the other columns. This is why we need to orient it, and transpose it.
                padded_series = pad_dict_list(series[0], False)
                frame = pd.DataFrame(padded_series)
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
    g.figure.suptitle(folder_name)
    # g.map_dataframe(annotate_with_stopping_line)
    for ax in g.axes.flat:
        # Get the augmentation method associated with this subplot
        augmentation_method = ax.get_title().split("=")[1].strip()

        # Filter the data for the current augmentation method
        subset_data = frame[frame[QUERY_STRATEGY_COLUMN] == augmentation_method]

        for criterion in STOPPING_CRITERIA:
            if subset_data[criterion].any():
                # Find the first iteration where stop_history is True
                first_true_iteration = subset_data.loc[
                    subset_data[criterion]
                ].iloc[0]["iterations"]

                # Add a vertical line at the first true iteration
                linestyle = (
                    # Conservative = solid, Middle Ground = dashed, Aggressive = dotted
                    # If the criterion is not in the name, the line is dashdot
                    "solid"
                    if "conservative" in criterion
                    else "dashed"
                    if "middle_ground" in criterion
                    else "dotted"
                    if "aggressive" in criterion
                    else "dashdot"
                )
                ax.axvline(
                    x=first_true_iteration,
                    color="r",
                    linestyle=linestyle,
                    label=criterion,
                )

                # Add a text label indicating the first true iteration
                ax.text(
                    first_true_iteration,
                    0.5,
                    f"Iteration {first_true_iteration} / {criterion}",
                    color="r",
                    ha="right",
                    va="center",
                    rotation=90,
                    transform=ax.get_xaxis_transform(),
                )

    plt.tight_layout()
    plt.show()


create_graph_for_augmentation_type("None")
