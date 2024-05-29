import matplotlib.pyplot as plt
import seaborn as sns
from evaluation.constants import (
    AugmentationType,
    Datasets,
    STOPPING_CRITERIA,
    QUERY_STRATEGY_COLUMN,
    LATEX_IMAGES_PATH,
)
from evaluation.utils import create_complete_frame



def create_graph_for_augmentation_type(folder_name: str):
    frame, n_frames = create_complete_frame(folder_name)

    g = sns.relplot(
        data=frame,
        x="iterations",
        y="test_accuracies",
        col=QUERY_STRATEGY_COLUMN,
        kind="line",
    )
    g.figure.subplots_adjust(top=0.9)  # adjust the Figure in rp
    g.figure.suptitle(f"{folder_name}, Amount of runs: {n_frames}")
    # g.map_dataframe(annotate_with_stopping_line)
    for ax in g.axes.flat:
        # Get the augmentation method associated with this subplot
        augmentation_method = ax.get_title().split("=")[1].strip()

        # Filter the data for the current augmentation method
        subset_data = frame[frame[QUERY_STRATEGY_COLUMN] == augmentation_method]

        for criterion in STOPPING_CRITERIA:
            if subset_data[criterion].any():
                average_true_iteration = int(
                    subset_data.loc[subset_data[criterion] == True, "iterations"].mean()
                )
                # print(f"First true iteration for {criterion}: {first_true_iteration}")

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
                color = (
                    # Conservative = solid, Middle Ground = dashed, Aggressive = dotted
                    # If the criterion is not in the name, the line is dashdot
                    "r"
                    if "kappa" in criterion
                    else "g"
                    if "classification" in criterion
                    else "b"
                    if "delta" in criterion
                    else "y"
                )
                label = (
                    # Conservative = solid, Middle Ground = dashed, Aggressive = dotted
                    # If the criterion is not in the name, the line is dashdot
                    "Kappa Average"
                    if "kappa" in criterion
                    else "Classification Change"
                    if "classification" in criterion
                    else "Delta F-Score"
                    if "delta" in criterion
                    else "y"
                )
                label = (
                    label
                    + " "
                    + (
                        "Conservative"
                        if "conservative" in criterion
                        else "Middle Ground"
                        if "middle_ground" in criterion
                        else "Aggressive"
                        if "aggressive" in criterion
                        else ""
                    )
                )
                ax.axvline(
                    x=average_true_iteration,
                    color=color,
                    linestyle=linestyle,
                    label=label,
                )

                # Add a text label indicating the first true iteration
                ax.text(
                    average_true_iteration,
                    0.5,
                    f"Iteration {average_true_iteration}",
                    color=color,
                    ha="right",
                    va="center",
                    rotation=90,
                    transform=ax.get_xaxis_transform(),
                )

                ax.legend(loc=0)

    plt.tight_layout()
    plt.savefig(f"{LATEX_IMAGES_PATH}{folder_name}.png")
    # plt.show()


create_graph_for_augmentation_type(f"None/{Datasets.TWEET.value}")
create_graph_for_augmentation_type(f"None/{Datasets.IMDB.value}")
create_graph_for_augmentation_type(f"None/{Datasets.AG_NEWS.value}")

create_graph_for_augmentation_type(f"{AugmentationType.BACK_TRANSLATION_AUG.value}/{Datasets.TWEET.value}")
create_graph_for_augmentation_type(
    f"{AugmentationType.BACK_TRANSLATION_AUG.value}/{Datasets.IMDB.value}"
)
create_graph_for_augmentation_type(f"{AugmentationType.BACK_TRANSLATION_AUG.value}/{Datasets.AG_NEWS.value}")

create_graph_for_augmentation_type(
    f"{AugmentationType.SYNONYM_AUG.value}/{Datasets.TWEET.value}"
)
create_graph_for_augmentation_type(
    f"{AugmentationType.SYNONYM_AUG.value}/{Datasets.IMDB.value}"
)
create_graph_for_augmentation_type(f"{AugmentationType.SYNONYM_AUG.value}/{Datasets.AG_NEWS.value}")

create_graph_for_augmentation_type(f"{AugmentationType.CONTEXTUAL_WORD_EMBS.value}/{Datasets.TWEET.value}")
create_graph_for_augmentation_type(
    f"{AugmentationType.CONTEXTUAL_WORD_EMBS.value}/{Datasets.IMDB.value}"
)
create_graph_for_augmentation_type(f"{AugmentationType.CONTEXTUAL_WORD_EMBS.value}/{Datasets.AG_NEWS.value}")

create_graph_for_augmentation_type(
    f"{AugmentationType.RANDOM_SWAP.value}/{Datasets.TWEET.value}"
)
create_graph_for_augmentation_type(
    f"{AugmentationType.RANDOM_SWAP.value}/{Datasets.IMDB.value}"
)
create_graph_for_augmentation_type(f"{AugmentationType.RANDOM_SWAP.value}/{Datasets.AG_NEWS.value}")
