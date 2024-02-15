import copy
import datasets
import multiprocessing

from datasets import load_dataset, concatenate_datasets
import nlpaug.augmenter.word as naw

aug = naw.SynonymAug(aug_src="wordnet")


def create_augmented_dataset(
    dataset: datasets.Dataset, augmenter: naw.Augmenter, feature: str = "text", n=3
) -> set[datasets.Dataset, dict[int, list[int]]]:
    augmented_indices = {}

    num_rows = dataset.num_rows
    # TODO: This calculation is wrong 
    for x in range(num_rows):
        augmented_indices[int(x)] = [
            int((x + num_rows) * (y + 1)) for y in range(n)
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

# n = 5
# dset = datasets.load_dataset("glue", "mrpc", split="train").select(
#     [0, 10, 20, 30, 40, 50]
# )

# augmented_sets = []

# for x in range(n):
#     augmented_sets.append(dset.map(aug))


# def aug_set(example, feature: str = "text"):
#     example["sentence1"] = aug.augment(example["sentence1"])[0]

#     return example


# def augment_hf_set(dataset: datasets.Dataset, augmenter):
#     print(dataset["sentence1"][:5])
#     updated_dataset = dataset.map(aug_set)
#     print(updated_dataset["sentence1"][:5])


# def aug(samples, index):
#     # Simply copy the existing data to have x2 amount of data
#     print(samples)
#     print(index)
#     for k, v in samples.items():
#         samples[k].extend(v)
#     return samples


# small_dataset = datasets.load_dataset("glue", "mrpc", split="train").select(
#     [0, 10, 20, 30, 40, 50]
# )
# dataset = small_dataset.map(aug, batched=True, batch_size=2, with_indices=True)
# # print(dataset["sentence1"][:5])


# # small_dataset = datasets.load_dataset("glue", "mrpc", split="train").select([0, 10, 20, 30, 40, 50])
# # augment_hf_set(small_dataset, aug)

# # augmented_text = aug.augment(text)
# # print("Original:")
# # print(text)
# # print("Augmented Text:")
# # print(augmented_text)
