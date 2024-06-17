from utils import create_complete_frame
from constants import AugmentedPaths, BasePaths, LATEX_TABLES_RAW_PATH


def create_raw_table(folder_path: str):
    try:
        frame, _ = create_complete_frame(folder_path.value)
        frame.drop(columns=["samples_count"], inplace=True)

        for series_name, _ in frame.items():
            frame.rename(
                columns={series_name: series_name.replace("_", "\_")}, inplace=True
            )
        frame.to_latex(
            f"{LATEX_TABLES_RAW_PATH}/{folder_path.value.replace('/', '_')}.tex",
            index=False,
            float_format="{:.5f}".format,
        )
        print(frame)
    except FileNotFoundError:
        print(f"Could not find {folder_path.value}")


for folder_path in AugmentedPaths:
    create_raw_table(folder_path)
for folder_path in BasePaths:
    create_raw_table(folder_path)
