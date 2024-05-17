from pathlib import Path
import multiprocessing
import json

import datasets
import nlpaug
import nlpaug.augmenter.word as naw
from datasets import concatenate_datasets

aug = naw.SynonymAug(aug_src="wordnet")


def create_augmented_dataset(
    dataset: datasets.Dataset,
    augmenter,
    feature: str = "text",
    n: int = 3,
    saving_path: str = None,
) -> set[datasets.Dataset, dict[int, list[int]]]:
    augmented_indices = {}

    # Because we multiply create augmented sets
    # based on one set and then concatenate the newly
    # created virtual sets, the augmented_indices
    # always follow this pattern.
    num_rows = dataset.num_rows
    for x in range(num_rows):
        augmented_indices[int(x)] = [int(x + (num_rows * (y + 1))) for y in range(n)]

    augmented_sets = [
        dataset.map(
            lambda row: {feature: augmenter.augment(row[feature])[0]},
            # num_proc=4, # multiprocessing.cpu_count(),
        )
        for _ in range(n)
    ]
    augmented_sets.insert(0, dataset)

    augmented_full_set = concatenate_datasets(augmented_sets)

    augmenter_str = str(augmenter).replace(" ", "_")
    file_name = f"{augmenter_str}_{n}.txt"

    with open((Path(__file__).parent / "../results" / file_name).resolve(), "a") as f:
        f.write(
            "".join(
                [f"{augmented_full_set[i]}\n" for i in range(0, num_rows * n, num_rows)]
            )
        )

    augmented_full_set.save_to_disk(saving_path)
    with open(f"{saving_path}/augmented_indices.json", "w") as f:
        f.write(json.dump(augmented_indices))

    return augmented_full_set, augmented_indices
