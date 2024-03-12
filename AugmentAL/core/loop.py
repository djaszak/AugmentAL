import nlpaug.augmenter.word as naw
import numpy as np
import pandas as pd
from pathlib import Path
from core.augment import create_augmented_dataset
from core.constants import Datasets
from core.core import (
    create_active_learner,
    create_small_text_dataset,
    evaluate,
)
from core.query_strategies import (
    AugmentedLeastConfidenceQueryStrategy,
    AugmentedOutcomesQueryStrategy,
    AugmentedSearchSpaceExtensionAndOutcomeQueryStrategy,
    AugmentedSearchSpaceExtensionQueryStrategy,
)
from datasets import load_dataset
from small_text import BreakingTies, QueryStrategy

# CONSTANTS
SEED = 2022
np.random.seed(SEED)


def run_active_learning_loop(
    raw_test,
    raw_train,
    raw_augmented_train,
    augmented_indices,
    num_queries: int = 5,
    num_samples: int = 20,
    num_augmentations: int = 2,
    dataset: Datasets = Datasets.ROTTEN,
    query_strategy: str
    | QueryStrategy = "AugmentedSearchSpaceExtensionAndOutcomeQueryStrategy",
):
    num_classes = raw_train.features["label"].num_classes

    base_strategy = BreakingTies()
    augmented_least_confidence_strategy = AugmentedLeastConfidenceQueryStrategy(
        augmented_indices=augmented_indices
    )
    # Here we could add more custom configurations for the query strategies
    query_strategies: dict[str, QueryStrategy] = {
        "BreakingTies": BreakingTies(),
        "AugmentedSearchSpaceExtensionAndOutcomeQueryStrategy": AugmentedSearchSpaceExtensionAndOutcomeQueryStrategy(
            base_strategy=base_strategy, augmented_indices=augmented_indices
        ),
        "AugmentedSearchSpaceExtensionQueryStrategy": AugmentedSearchSpaceExtensionQueryStrategy(
            base_strategy=base_strategy, augmented_indices=augmented_indices
        ),
        "AugmentedOutcomesQueryStrategy": AugmentedOutcomesQueryStrategy(
            base_strategy=base_strategy, augmented_indices=augmented_indices
        ),
        "AugmentedSearchSpaceExtensionAndOutcomeLeastConfidenceQueryStrategy": AugmentedSearchSpaceExtensionAndOutcomeQueryStrategy(
            base_strategy=augmented_least_confidence_strategy, augmented_indices=augmented_indices
        ),
        "AugmentedSearchSpaceExtensionLeastConfidenceQueryStrategy": AugmentedSearchSpaceExtensionQueryStrategy(
            base_strategy=augmented_least_confidence_strategy, augmented_indices=augmented_indices
        ),
        "AugmentedOutcomesLeastConfidenceQueryStrategy": AugmentedOutcomesQueryStrategy(
            base_strategy=augmented_least_confidence_strategy,
            augmented_indices=augmented_indices,
        ),
    }

    test = create_small_text_dataset(raw_test)
    train = create_small_text_dataset(
        raw_augmented_train if query_strategy != "BreakingTies" else raw_train
    )
    chosen_strategy = (
        query_strategies[query_strategy]
        if isinstance(query_strategy, str)
        else query_strategy
    )
    active_learner, indices_labeled = create_active_learner(
        train_set=train,
        num_classes=num_classes,
        query_strategy=chosen_strategy,
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
        txt_filename = f"{query_strategy}_{base_strategy}_{num_queries}_queries_num_samples_{num_samples}_num_augmentations_{num_augmentations}.txt"
        indices_queried = active_learner.query(num_samples=num_samples)

        y = train.y[indices_queried]

        # Return the labels for the current query to the active learner.
        active_learner.update(y)

        indices_labeled = np.concatenate([indices_queried, indices_labeled])

        print("---------------")
        print(f"Iteration #{i} ({len(indices_labeled)} samples)")
        results.append(evaluate(active_learner, train[indices_labeled], test))

        # Write indices_queried to a txt file after every third iteration
        if (i + 1) % 3 == 0:
            with open(
                (Path(__file__).parent / "../results" / txt_filename).resolve(), "a"
            ) as f:
                f.write(", ".join(map(str, indices_queried)) + "\n")

    iterations = np.arange(num_queries + 1)
    accuracies = np.array(results)

    # convert to pandas dataframe
    d = {"iterations": iterations, "accuracies": accuracies}
    d_frame = pd.DataFrame(d)

    saving_name = f"{query_strategy}_{base_strategy}_{num_queries}_queries_num_samples_{num_samples}_num_augmentations_{num_augmentations}.json"

    with open((Path(__file__).parent / "../results" / saving_name).resolve(), "w") as f:
        f.write(d_frame.to_json())
