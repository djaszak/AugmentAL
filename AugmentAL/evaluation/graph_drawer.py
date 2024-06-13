import matplotlib.pyplot as plt
import seaborn as sns
from constants import (
    AugmentationType,
    LATEX_IMAGES_PATH,
    BaselineStrategies,
    AugmentedStrategies,
)
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
            "tweet_eval": "Tweet Eval Hate",
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
        },
        # legend=False,
    )

    # Adjusting subplot parameters to reduce overlap
    g.figure.subplots_adjust(wspace=0.1, hspace=0.2)
    g.set_axis_labels("Iterations", "Test Accuracy")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")

    # Overlay baseline data on each subplot
    baseline_colors = ['gray', 'blue']  # Define different colors for baselines
    for (row_val, _), ax in g.axes_dict.items():
        # Filtering the baseline data for the current subplot's dataset
        subset = baseline_data[baseline_data["Dataset"] == row_val]

        if not subset.empty:
            # Plotting the baseline data for each Query Strategy in the baseline data
            for i, query_strategy in enumerate(subset["Query Strategy"].unique()):
                baseline_subset = subset[subset["Query Strategy"] == query_strategy]
                sns.lineplot(
                    data=baseline_subset,
                    x="iterations",
                    y="test_accuracies",
                    ax=ax,
                    label=f"Baseline ({query_strategy})",
                    color=baseline_colors[i % len(baseline_colors)],
                    linestyle='--',
                    # legend=False
                )

    # Adjusting the legend
    if g.axes.shape[0] > 1 and g.axes.shape[1] > 1:
        handles, labels = g.axes[0, 0].get_legend_handles_labels()
        g.legend.remove()
    # g.add_legend(title='Query Strategy', labels=labels, handles=handles, loc='upper left')
    
    # Saving the plot
    plt.tight_layout()
    plt.savefig(f"{LATEX_IMAGES_PATH}/full_comparison.png")
    plt.show()

create_graph_for_augmentation_type()
