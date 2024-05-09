import numpy as np
from pathlib import Path
from core.core import (
    create_active_learner,
    create_small_text_dataset,
    evaluate,
)
from core.constants import TransformerModels
from core.query_strategies import (
    AugmentedOutcomesQueryStrategy,
    AugmentedSearchSpaceExtensionAndOutcomeQueryStrategy,
    AugmentedSearchSpaceExtensionQueryStrategy,
    AverageAcrossAugmentedQueryStrategy,
)
from small_text import BreakingTies, QueryStrategy, KappaAverage, RandomSampling

# CONSTANTS
SEED = 2022
np.random.seed(SEED)


def run_active_learning_loop(
    raw_test,
    raw_train,
    augmented_indices,
    num_queries: int = 5,
    num_samples: int = 20,
    num_augmentations: int = 2,
    query_strategy: str
    | QueryStrategy = "AugmentedSearchSpaceExtensionAndOutcomeQueryStrategy",
    base_strategy: QueryStrategy = BreakingTies(),
    model: str = TransformerModels.BERT_TINY.value,
    device: str = "",
) -> (dict, str):
    num_classes = raw_train.features["label"].num_classes
    # Use the default KappaAverage parameters that follow
    # the results from the original paper.
    # kappa=0.9 and window_size=3
    stopping_criterion = KappaAverage(num_classes)

    average_across_augmented_strategy = AverageAcrossAugmentedQueryStrategy(
        base_strategy=base_strategy, augmented_indices=augmented_indices
    )
    # Here we could add more custom configurations for the query strategies
    query_strategies: dict[str, QueryStrategy] = {
        "RandomSampling": RandomSampling(),
        "BreakingTies": base_strategy,
        "AugmentedSearchSpaceExtensionQueryStrategy": AugmentedSearchSpaceExtensionQueryStrategy(
            base_strategy=base_strategy, augmented_indices=augmented_indices
        ),
        "AugmentedSearchSpaceExtensionAndOutcomeQueryStrategy": AugmentedSearchSpaceExtensionAndOutcomeQueryStrategy(
            base_strategy=base_strategy, augmented_indices=augmented_indices
        ),
        "AugmentedOutcomesQueryStrategy": AugmentedOutcomesQueryStrategy(
            base_strategy=base_strategy, augmented_indices=augmented_indices
        ),
        "AverageAcrossAugmentedQueryStrategy": average_across_augmented_strategy,
        "AverageAcrossAugmentedExtendedOutcomesQueryStrategy": AugmentedSearchSpaceExtensionAndOutcomeQueryStrategy(
            base_strategy=average_across_augmented_strategy,
            augmented_indices=augmented_indices,
        ),
    }

    test = create_small_text_dataset(raw_test)
    train = create_small_text_dataset(raw_train)
    chosen_strategy = (
        query_strategies[query_strategy]
        if isinstance(query_strategy, str)
        else query_strategy
    )
    active_learner, indices_labeled = create_active_learner(
        train_set=train,
        num_classes=num_classes,
        training_indices=[x for x in augmented_indices.keys()]
        if augmented_indices
        else None,
        query_strategy=chosen_strategy,
        model=model,
        device=device,
    )

    test_results = []
    train_results = []
    stopping_history = []
    samples_count = [len(indices_labeled)]

    train_results.append(evaluate(active_learner, train[indices_labeled], test)[0])
    test_results.append(evaluate(active_learner, train[indices_labeled], test)[1])

    stopping_history.append(
        stopping_criterion.stop(predictions=active_learner.classifier.predict(train))
    )

    for i in range(num_queries):
        # ...where each iteration consists of labelling 20 samples
        # Using the AugmentedExpansionQueryStrategy will result in more than 20
        # samples provided. This has to be handled by usage of
        # augmented_indices when implementing a real life usage.
        # At the end you should always have 20 original indices in this
        # list. Just iterate over it, let the original sample be labeled
        # and add the label that is gotten to the virtual samples, which
        # are retrieved by augmented_indices.
        txt_filename = f"{query_strategy}_{base_strategy}_{num_queries}_queries_{num_samples}_num_samples_{num_augmentations}_num_augmentations.txt"
        indices_queried = active_learner.query(num_samples=num_samples)

        y = train.y[indices_queried]

        # Return the labels for the current query to the active learner.
        active_learner.update(y)

        indices_labeled = np.concatenate([indices_queried, indices_labeled])

        print("---------------")
        print(f"Iteration #{i} ({len(indices_labeled)} samples)")
        train_results.append(evaluate(active_learner, train[indices_labeled], test)[0])
        test_results.append(evaluate(active_learner, train[indices_labeled], test)[1])

        stopping_criterion_response = stopping_criterion.stop(
            predictions=active_learner.classifier.predict(train)
        )
        print(f"Stop: {stopping_criterion_response}")
        stopping_history.append(stopping_criterion_response)

        # Write indices_queried to a txt file after every third iteration
        if (i + 1) % 3 == 0:
            with open(
                (Path(__file__).parent / "../results" / txt_filename).resolve(), "a"
            ) as f:
                f.write(", ".join(map(str, indices_queried)) + "\n")

    iterations = np.arange(num_queries + 1)
    test_accuracies = np.array(test_results)
    train_accuracies = np.array(train_results)

    # convert to pandas dataframe
    d = {
        "iterations": iterations,
        "test_accuracies": test_accuracies,
        "train_accuracies": train_accuracies,
        "stopping_history": stopping_history,
        "samples_count": samples_count,
    }

    details_str = f"{query_strategy}_{base_strategy}_{num_queries}_queries_num_samples_{num_samples}_num_augmentations_{num_augmentations}"

    # with open((Path(__file__).parent / "../results" / saving_name).resolve(), "w") as f:
    # f.write(d_frame.to_json())

    return d, details_str
