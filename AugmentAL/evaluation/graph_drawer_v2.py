import matplotlib.pyplot as plt
import numpy as np
import os
import json
import pandas as pd
from pathlib import Path

def get_json_files(folder_path):
    """
    Get all JSON files from a given folder.

    Args:
    - folder_path (str): Path to the folder.

    Returns:
    - list: List of JSON files.
    """
    # json_files = []
    # for file_name in os.listdir(folder_path):
    #     file_path = os.path.join(folder_path, file_name)
    #     if os.path.isfile(file_path) and file_name.endswith(".json"):
    #         json_files.append(file_path)
    # return json_files

    files = []

    root_dir = Path(__file__).parent / "../results"

    for subdir, dirs, files in os.walk(root_dir):
        print(len(files))
        for file in files:
            if ".json" in file:
                files.append(os.path.join(subdir, file))

    return files


def create_graph_for_augmentation_type(augmentation_type):
    for file in get_json_files(str(Path(__file__).parent / "../results")):
        if augmentation_type in file:
            with open(file, "r") as f:
                frame = (
                    pd.json_normalize(json.load(f)["test_accuracies"])
                    .melt(id_vars=["0"], var_name="Iteration", value_name="Accuracy")
                    .drop(columns=["0"])
                )

                plt.plot(
                    frame["Iteration"],
                    frame["Accuracy"],
                    label=file.split("/")[-1]
                    + " (mean: "
                    + str(round(frame["Accuracy"].mean(), 2))
                    + ")",
                )

    plt.legend()
    plt.show(block=True)

create_graph_for_augmentation_type("BackTranslationAug")





# for file in get_json_files(str(Path(__file__).parent / "../results")):
#     print(file)