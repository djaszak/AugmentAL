import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from constants import (
    AugmentationType,
    LATEX_IMAGES_PATH,
    BaselineStrategies,
    AugmentedStrategies,
)
import constants
from utils import create_complete_frame_for_all_folders


def create_graph_for_augmentation_type():
    frame, n_frames = create_complete_frame_for_all_folders()

    # Replacing values in the DataFrame for readability
    frame.replace(
        {
            AugmentationType.BACK_TRANSLATION_AUG.value: "Backtranslation",
            AugmentationType.SYNONYM_AUG.value: "Synonym",
            AugmentationType.RANDOM_SWAP.value: "Random Swap",
            AugmentationType.CONTEXTUAL_WORD_EMBS.value: "BERT",
            "tweet_eval": "Tweet",
            "imdb": "IMDB",
            "ag_news": "AG News",
            BaselineStrategies.RANDOM_SAMPLING.value: "Random Sampling",
            BaselineStrategies.BREAKING_TIES.value: "Breaking Ties",
            AugmentedStrategies.AUGMENTED_OUTCOME.value: "Extended Outcome",
            AugmentedStrategies.AUGMENTED_SEARCH_SPACE.value: "Extended Search Space",
            AugmentedStrategies.AVERAGE_ACROSS_AUGMENTED.value: "AAA",
        },
        inplace=True,
    )

    # Renaming columns for readability
    frame.rename(
        columns={
            "dataset": "Dataset",
            "augmentation_method": "Augmentation Method",
            "query_strategy": "Query Strategy",
        },
        inplace=True,
    )

    # Separate the baseline data
    baseline_data = frame[frame["Augmentation Method"] == "None"]
    plot_data = frame[frame["Augmentation Method"] != "None"]

    # Create the relplot for the main data
    font = {
        # "weight": "bold",
        "size": constants.FONT_SIZE,
    }

    matplotlib.rc("font", **font)
    # sns.set_theme(rc={'figure.figsize':(20,10)})
    g = sns.relplot(
        data=plot_data,
        x="iterations",
        y="test_accuracies",
        col="Augmentation Method",
        kind="line",
        hue="Query Strategy",
        row="Dataset",
        facet_kws={
            "margin_titles": True,
            "legend_out": True,
        },
        # legend_out=True,
        # height=3,
    )
    # g.axes.legend(loc='upper left', fontsize=20,bbox_to_anchor=(0, 1.1))
    # g.axes.set_xlabel("Iterations", fontsize=constants.FONT_SIZE,)
    # g.axes.set_ylabel("Test Accuracy", fontsize=constants.FONT_SIZE,)

    for ax in g.axes.flat:
        ax.tick_params(axis="both", which="major", labelsize=constants.FONT_SIZE)
        ax.tick_params(axis="both", which="minor", labelsize=constants.FONT_SIZE)
        ax.set_xlabel(
            "Iterations",
            fontsize=constants.FONT_SIZE,
        )
        ax.set_ylabel(
            "Test Accuracy",
            fontsize=constants.FONT_SIZE,
        )

    # Adjusting subplot parameters to reduce overlap
    # g.figure.subplots_adjust(wspace=0.1, hspace=0.2)
    # g.set_axis_labels(, "Test Accuracy")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    # sns.move_legend(
    #     g,
    #     "upper center",
    #     bbox_to_anchor=(0.5, 1),
    #     ncol=5,
    #     title=None,
    #     frameon=False,
    # )

    # Saving the plot
    # plt.tight_layout()
    # plt.figure(dpi=300)
    plt.savefig(f"{LATEX_IMAGES_PATH}/full_comparison.png")
    # plt.show()


create_graph_for_augmentation_type()
