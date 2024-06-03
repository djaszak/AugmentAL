import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from constants import (
    AugmentationType,
    Datasets,
    STOPPING_CRITERIA,
    QUERY_STRATEGY_COLUMN,
    LATEX_IMAGES_PATH,
    AugmentedPaths,
    BasePaths

)
from utils import create_complete_frame


def create_graph_for_augmentation_type(folder_name: str):
    frame, n_frames = create_complete_frame(folder_name)

    g = sns.relplot(
        data=frame,
        x="iterations",
        y="test_accuracies",
        col=QUERY_STRATEGY_COLUMN,
        kind="line",
    )
    g.set_axis_labels("Iterations", "Test Accuracy")
    # g.xaxis.set_major_locator(ticker.MultipleLocator(5))
    # g.xaxis.set_major_formatter(ticker.ScalarFormatter())
    g.figure.subplots_adjust(top=0.9)  # adjust the Figure in rp
    g.figure.suptitle(f"{folder_name}, Amount of runs: {n_frames}")
    # g.refline(y=0.5, linestyle="--", color="r")
    # g.refline(y=frame["test_accuracies"].max(), linestyle="--", color="r")
    # g.map_dataframe(annotate_with_stopping_line)
    for ax in g.axes.flat:
        # Get the augmentation method associated with this subplot
        augmentation_method = ax.get_title().split("=")[1].strip()

        # Filter the data for the current augmentation method
        subset_data = frame[frame[QUERY_STRATEGY_COLUMN] == augmentation_method]

        

        # Add horizontal line at highest mean accuracy
        highest_mean_accuracy = subset_data["test_accuracies"].max()

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
                    "KA"
                    if "kappa" in criterion
                    else "CC"
                    if "classification" in criterion
                    else "DF"
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
                label += f" Iter: {average_true_iteration}, Acc: {round(subset_data.loc[subset_data['iterations'] == average_true_iteration, 'test_accuracies'].mean(), 2)}"
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
                    f"",
                    color=color,
                    ha="right",
                    va="center",
                    rotation=90,
                    transform=ax.get_xaxis_transform(),
                )

                ax.legend(loc=0)

    # g.set(xticks=x_ticks)
    plt.tight_layout()
    plt.savefig(f"{LATEX_IMAGES_PATH}{folder_name}.png")
    # plt.show()

for folder_path in AugmentedPaths:
    try:
        create_graph_for_augmentation_type(folder_path.value)
    except FileNotFoundError:
        continue
for folder_path in BasePaths:
    try:
        create_graph_for_augmentation_type(folder_path.value)
    except FileNotFoundError:
        continue

# create_graph_for_augmentation_type(f"None/{Datasets.TWEET.value}")
# create_graph_for_augmentation_type(f"None/{Datasets.IMDB.value}")
# create_graph_for_augmentation_type(f"None/{Datasets.AG_NEWS.value}")

# create_graph_for_augmentation_type(f"{AugmentationType.BACK_TRANSLATION_AUG.value}/{Datasets.TWEET.value}")
# create_graph_for_augmentation_type(
#     f"{AugmentationType.BACK_TRANSLATION_AUG.value}/{Datasets.IMDB.value}"
# )
# create_graph_for_augmentation_type(f"{AugmentationType.BACK_TRANSLATION_AUG.value}/{Datasets.AG_NEWS.value}")

# create_graph_for_augmentation_type(
#     f"{AugmentationType.SYNONYM_AUG.value}/{Datasets.TWEET.value}"
# )
# create_graph_for_augmentation_type(
#     f"{AugmentationType.SYNONYM_AUG.value}/{Datasets.IMDB.value}"
# )
# create_graph_for_augmentation_type(f"{AugmentationType.SYNONYM_AUG.value}/{Datasets.AG_NEWS.value}")

# create_graph_for_augmentation_type(f"{AugmentationType.CONTEXTUAL_WORD_EMBS.value}/{Datasets.TWEET.value}")
# create_graph_for_augmentation_type(
#     f"{AugmentationType.CONTEXTUAL_WORD_EMBS.value}/{Datasets.IMDB.value}"
# )
# create_graph_for_augmentation_type(f"{AugmentationType.CONTEXTUAL_WORD_EMBS.value}/{Datasets.AG_NEWS.value}")

# create_graph_for_augmentation_type(
#     f"{AugmentationType.RANDOM_SWAP.value}/{Datasets.TWEET.value}"
# )
# create_graph_for_augmentation_type(
#     f"{AugmentationType.RANDOM_SWAP.value}/{Datasets.IMDB.value}"
# )
# create_graph_for_augmentation_type(f"{AugmentationType.RANDOM_SWAP.value}/{Datasets.AG_NEWS.value}")
