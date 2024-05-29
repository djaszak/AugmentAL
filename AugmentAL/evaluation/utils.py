import os

from pathlib import Path
import pandas as pd
from evaluation.constants import QUERY_STRATEGY_COLUMN

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


def pad_dict_list(dict_list, padel):
    lmax = 0
    for lname in dict_list.keys():
        lmax = max(lmax, len(dict_list[lname]))
    for lname in dict_list.keys():
        ll = len(dict_list[lname])
        if ll < lmax:
            dict_list[lname] += [padel] * (lmax - ll)
    return dict_list


def create_complete_frame(folder_name: str) -> tuple[pd.DataFrame, int]:
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

    return pd.concat(frames), len(frames)

