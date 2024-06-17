import pandas as pd
from constants import (
    STOPPING_CRITERIA,
    BasePaths,
    AugmentedPaths,
    AugmentedStrategies,
    AUGMENTATION_METHOD_VERBOSE,
    DATASETS_VERBOSE,
    STOPPING_CRITERIA_VERBOSE,
)
from utils import get_query_strategy_frame

results_dict = {
    "Stopping Criterion": [],
    "Augmentation Method": [],
    "Average Iteration": [],
    "Average Accuracy": [],
    "Dataset": [],
}

# Define the column names for the dataframe
QUERY_STRATEGY_COLUMN = "query_strategy"
ITERATION_COLUMN = "iterations"
ACCURACY_COLUMN = "test_accuracies"

for folder_path in AugmentedPaths:
    augmentation_method = folder_path.value.split("/")[0]
    dataset = folder_path.value.split("/")[-1]

    # Baseline folder path
    baseline_folder = f"None/{dataset}"

    for strategy in AugmentedStrategies:
        # Load the augmented strategy frame
        frame = get_query_strategy_frame(folder_path.value, strategy.value)

        if frame.empty:
            continue

        for criterion in STOPPING_CRITERIA:
            subset_data = frame[frame[QUERY_STRATEGY_COLUMN] == strategy.value]

            if subset_data[criterion].any():
                # Calculate the average iteration
                average_true_iteration = int(
                    subset_data.loc[
                        subset_data[criterion] == True, ITERATION_COLUMN
                    ].mean()
                )

                # Calculate the average accuracy
                average_accuracy = subset_data.loc[
                    subset_data[criterion] == True, ACCURACY_COLUMN
                ].mean()

                # Update the results dictionary
                results_dict["Stopping Criterion"].append(
                    STOPPING_CRITERIA_VERBOSE[criterion]
                )
                results_dict["Augmentation Method"].append(
                    AUGMENTATION_METHOD_VERBOSE[augmentation_method]
                )
                results_dict["Average Iteration"].append(average_true_iteration)
                results_dict["Average Accuracy"].append(average_accuracy)
                results_dict["Dataset"].append(DATASETS_VERBOSE[dataset])

# Convert results dictionary to DataFrame
results_df = pd.DataFrame(results_dict)

# Generate LaTeX table
latex_table = results_df.to_latex(index=False, float_format="%.2f")
print(latex_table)
