import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nlpaug.augmenter.word as naw

from small_text import BreakingTies
from augment import create_augmented_dataset
from datasets import load_dataset
from helpers import evaluate, create_active_learner, create_small_text_dataset
from constants import Datasets
from query_strategies import (
    AugmentedSearchSpaceExtensionAndOutcomeQueryStrategy,
    AugmentedSearchSpaceExtensionQueryStrategy,
    AugmentedLeastConfidenceQueryStrategy,
    AugmentedOutcomesQueryStrategy,
)

# SETUP

num_queries = 5
num_samples = 20
num_augmentations = 2

dataset = load_dataset(Datasets.ROTTEN.value)

raw_test = dataset["test"]
raw_train = dataset["train"]
num_classes = raw_train.features["label"].num_classes
raw_augmented_train, augmented_indices = create_augmented_dataset(
    raw_train, naw.SynonymAug(aug_src="wordnet"), n=num_augmentations
)

test = create_small_text_dataset(raw_test)
train = create_small_text_dataset(raw_augmented_train)
base_train = create_small_text_dataset(raw_train)
# train = create_small_text_dataset(raw_train)

# HERE ONE COULD ADD ITS OWN QUERY_STRATEGY
# TODO: adjust warm starting so that no augmented samples are used
active_learner, indices_labeled = create_active_learner(
    train_set=train,
    num_classes=num_classes,
    # query_strategy=BreakingTies(),
    query_strategy=AugmentedSearchSpaceExtensionQueryStrategy(
        base_strategy=BreakingTies(), augmented_indices=augmented_indices
    ),
    # query_strategy=AugmentedOutcomesQueryStrategy(
    #     base_strategy=BreakingTies(), augmented_indices=augmented_indices,
    #     base_dataset=base_train, base_indices_unlabeled=base_train, base_indices_labeled=base_train
    # ),
    # query_strategy=AugmentedExtensionQueryStrategy(base_strategy=AugmentedEntropyQueryStrategy(), augmented_indices=augmented_indices),
    # query_strategy=AugmentedEntropyQueryStrategy(augmented_indices=augmented_indices),
)

# USE INITIALIZED ACTIVE LEARNER AND GO INTO LOOP
results = []
results.append(evaluate(active_learner, train[indices_labeled], test))


for i in range(num_queries):
    # ...where each iteration consists of labelling 20 samples
    # Using the AugmentedExpansionQueryStrategy will result in more than 20
    # samples provided. This has to be handled by usage of
    # augmented_indices when implementing a real life usage.
    # At the end you should always have 20 original indices in this
    # list. Just iterate over it, let the original sample be labeled
    # and add the label that is gotten to the virtual samples, which
    # are retrieved by augmented_indices.
    indices_queried = active_learner.query(num_samples=num_samples)

    y = train.y[indices_queried]

    # Return the labels for the current query to the active learner.
    active_learner.update(y)

    indices_labeled = np.concatenate([indices_queried, indices_labeled])

    print("---------------")
    print(f"Iteration #{i} ({len(indices_labeled)} samples)")
    results.append(evaluate(active_learner, train[indices_labeled], test))


fig = plt.figure(figsize=(12, 8))
ax = plt.axes()

iterations = np.arange(num_queries + 1)
accuracies = np.array(results)

# convert to pandas dataframe
d = {"iterations": iterations, "accuracies": accuracies}
d_frame = pd.DataFrame(d)

with open(f"breaking_ties_20_queries_results.json", "w") as f:
    f.write(d_frame.to_json())
