import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from constants import (
    AugmentationType,
    Datasets,
    STOPPING_CRITERIA,
    QUERY_STRATEGY_COLUMN,
    DATASET_COLUMN,
    AUGMENTATION_METHOD_COLUMN,
    LATEX_IMAGES_PATH,
    AugmentedPaths,
    BasePaths

)
from utils import create_complete_frame, create_complete_frame_for_all_folders


def create_graph_for_augmentation_type(folder_name: str):
    frame, n_frames = create_complete_frame_for_all_folders()

    frame.replace(
        AugmentationType.BACK_TRANSLATION_AUG.value,
        "Backtranslation",
        inplace=True,
    )
    frame.replace(
        AugmentationType.SYNONYM_AUG.value,
        "Synonym",
        inplace=True,
    )
    frame.replace(
        AugmentationType.RANDOM_SWAP.value,
        "Random Swap",
        inplace=True,
    )
    frame.replace(
        AugmentationType.CONTEXTUAL_WORD_EMBS.value,
        "BERT",
        inplace=True,
    )
    frame.replace(
        "tweet_eval",
        "Tweet Eval Hate",
        inplace=True,
    )
    frame.replace(
        "imdb",
        "IMDB",
        inplace=True,
    )
    frame.replace(
        "ag_news",
        "AG News",
        inplace=True,
    )
    frame.rename(
        columns={
            "dataset": "Dataset",
            "augmentation_method": "Augmentation Method",
            "query_strategy": "Query Strategy",
        },
        inplace=True,
    )
    # frame.replace(
    #     {
    #         AUGMENTATION_METHOD_COLUMN: AugmentationType.BACK_TRANSLATION_AUG.value,
    #     },
    #     "Backtranslation",
    #     inplace=True,
    # )
    # plt.figure(figsize=(6.4, 20))
    g = sns.relplot(
        data=frame,
        x="iterations",
        y="test_accuracies",
        col="Augmentation Method",
        kind="line",
        hue="Query Strategy",
        row="Dataset",
        facet_kws={
            "margin_titles": True,
            # "despine": False,
        }
    )
    g.figure.subplots_adjust(wspace=0, hspace=0)
    g.set_axis_labels("Iterations", "Test Accuracy")
    g.figure.subplots_adjust(top=0.9)  # adjust the Figure in rp
    plt.tight_layout()
    plt.savefig(f"{LATEX_IMAGES_PATH}/full_comparison.png")
    # plt.show()

create_graph_for_augmentation_type("None/TWEET")

# for folder_path in AugmentedPaths:
#     try:
#         create_graph_for_augmentation_type(folder_path.value)
#     except FileNotFoundError:
#         continue
# for folder_path in BasePaths:
#     try:
#         create_graph_for_augmentation_type(folder_path.value)
#     except FileNotFoundError:
#         continue
