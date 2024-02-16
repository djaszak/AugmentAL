import datasets
import multiprocessing

from datasets import concatenate_datasets
import nlpaug.augmenter.word as naw

aug = naw.SynonymAug(aug_src="wordnet")


def create_augmented_dataset(
    dataset: datasets.Dataset, augmenter: naw.Augmenter, feature: str = "text", n=3
) -> set[datasets.Dataset, dict[int, list[int]]]:
    augmented_indices = {}

    # Because we multiply create augmented sets
    # based on one set and then concatenate the newly
    # created virtual sets, the augmented_indices
    # always follow this pattern.
    num_rows = dataset.num_rows
    for x in range(num_rows):
        augmented_indices[int(x)] = [
            int(x + (num_rows * (y + 1))) for y in range(n)
        ]

    augmented_sets = [
            dataset.map(
                lambda row: {feature: augmenter.augment(row[feature])[0]},
                # num_proc=multiprocessing.cpu_count(),
            )
            for _ in range(n)
        ]
    augmented_sets.insert(0, dataset)

    return concatenate_datasets(augmented_sets), augmented_indices
