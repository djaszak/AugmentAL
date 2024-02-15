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
    AugmentedEntropyQueryStrategy,
    AugmentedExtensionQueryStrategy,
)

# SETUP

dataset = load_dataset(Datasets.ROTTEN.value)

raw_test = dataset["test"]
raw_train = dataset["train"]
num_classes = raw_train.features["label"].num_classes
augmented_train, augmented_indices = create_augmented_dataset(
    raw_train, naw.SynonymAug(aug_src="wordnet"), n=2
)

test = create_small_text_dataset(raw_test)
train = create_small_text_dataset(augmented_train)

# HERE ONE COULD ADD ITS OWN QUERY_STRATEGY
active_learner, indices_labeled = create_active_learner(
    train,
    num_classes,
    AugmentedExtensionQueryStrategy(
        base_strategy=BreakingTies(), augmented_indices=augmented_indices
    ),
)

# USE INITIALIZED ACTIVE LEARNER AND GO INTO LOOP
num_queries = 10

results = []
results.append(evaluate(active_learner, train[indices_labeled], test))


for i in range(num_queries):
    # ...where each iteration consists of labelling 20 samples
    indices_queried = active_learner.query(num_samples=20)
    print(f"indices_queried: {indices_queried}")

    for index in indices_queried:
        if index in augmented_indices:
            indices_queried.append(augmented_indices[index])

    # Simulate user interaction here. Replace this for real-world usage.
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

with open(f"PredictionEntropy_results.json", "w") as f:
    f.write(d_frame.to_json())
