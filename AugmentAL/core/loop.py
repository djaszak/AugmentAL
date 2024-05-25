from datetime import datetime
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
from small_text import (
    BreakingTies,
    QueryStrategy,
    KappaAverage,
    RandomSampling,
    OverallUncertainty,
    DeltaFScore,
    ClassificationChange,
)

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
    # Define different stopping criteria, 4 ones are given by small-text
    # Every criterion will be configured in three variants
    # Conservative, which will be the default one
    # Middle ground, which will be a bit more aggressive
    # Aggressive, which will be the most aggressive one

    # Kappa Average
    kappa_average_conservative = KappaAverage(num_classes)
    kappa_average_middle_ground = KappaAverage(num_classes, kappa=0.9)
    kappa_average_aggressive = KappaAverage(num_classes, kappa=0.8)

    # Delta F-Score
    delta_f_score_conservative = DeltaFScore(num_classes)
    delta_f_score_middle_ground = DeltaFScore(num_classes, threshold=0.07)
    delta_f_score_aggressive = DeltaFScore(num_classes, threshold=0.09)

    # Classification Change
    classification_change_conservative = ClassificationChange(num_classes)
    classification_change_middle_ground = ClassificationChange(
        num_classes, threshold=0.04
    )
    classification_change_aggressive = ClassificationChange(num_classes, threshold=0.09)

    # Overall Uncertainty
    overall_uncertainty_conservative = OverallUncertainty(num_classes)
    overall_uncertainty_middle_ground = OverallUncertainty(num_classes, threshold=0.04)
    overall_uncertainty_aggressive = OverallUncertainty(num_classes, threshold=0.09)

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
    indices_train = [x for x in range(raw_train.num_rows)]
    test_results = []
    train_results = []
    stopping_history = []
    samples_count = [len(indices_labeled)]
    kappa_average_conservative_history = []
    kappa_average_middle_ground_history = []
    kappa_average_aggressive_history = []

    delta_f_score_conservative_history = []
    delta_f_score_middle_ground_history = []
    delta_f_score_aggressive_history = []

    classification_change_conservative_history = []
    classification_change_middle_ground_history = []
    classification_change_aggressive_history = []

    overall_uncertainty_conservative_history = []
    overall_uncertainty_middle_ground_history = []
    overall_uncertainty_aggressive_history = []

    train_results.append(evaluate(active_learner, train[indices_labeled], test)[0])
    test_results.append(evaluate(active_learner, train[indices_labeled], test)[1])

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

        # stopping_criterion_response = stopping_criterion.stop(
        #     predictions=active_learner.classifier.predict(train)
        # )
        # print(f"Stop: {stopping_criterion_response}")
        # stopping_history.append(stopping_criterion_response)

        stopping_criteria_start = datetime.now()
        print(f"Evaluating stopping criteria, starting at {stopping_criteria_start} \n")
        kappa_average_conservative_history.append(
            kappa_average_conservative.stop(
                predictions=active_learner.classifier.predict(train)
            )
        )
        kappa_average_middle_ground_history.append(
            kappa_average_middle_ground.stop(
                predictions=active_learner.classifier.predict(train)
            )
        )
        kappa_average_aggressive_history.append(
            kappa_average_aggressive.stop(
                predictions=active_learner.classifier.predict(train)
            )
        )

        delta_f_score_conservative_history.append(
            delta_f_score_conservative.stop(
                predictions=active_learner.classifier.predict(train)
            )
        )
        delta_f_score_middle_ground_history.append(
            delta_f_score_middle_ground.stop(
                predictions=active_learner.classifier.predict(train)
            )
        )
        delta_f_score_aggressive_history.append(
            delta_f_score_aggressive.stop(
                predictions=active_learner.classifier.predict(train)
            )
        )

        classification_change_conservative_history.append(
            classification_change_conservative.stop(
                predictions=active_learner.classifier.predict(train)
            )
        )
        classification_change_middle_ground_history.append(
            classification_change_middle_ground.stop(
                predictions=active_learner.classifier.predict(train)
            )
        )
        classification_change_aggressive_history.append(
            classification_change_aggressive.stop(
                predictions=active_learner.classifier.predict(train)
            )
        )

        # THIS WONT BE USED ANYMORE I THINK
        # indices_stopping = list(set(indices_train) - set(indices_labeled))
        # overall_uncertainty_conservative_history.append(
        #     overall_uncertainty_conservative.stop(
        #         predictions=active_learner.classifier.predict(train),
        #         indices_stopping=indices_stopping
        #     )
        # )
        # overall_uncertainty_middle_ground_history.append(
        #     overall_uncertainty_middle_ground.stop(
        #         predictions=active_learner.classifier.predict(train),
        #         indices_stopping=indices_stopping
        #     )
        # )
        # overall_uncertainty_aggressive_history.append(
        #     overall_uncertainty_aggressive.stop(
        #         predictions=active_learner.classifier.predict(train),
        #         indices_stopping=indices_stopping
        #     )
        # )
        stopping_criteria_end = datetime.now()
        print(f"Finished evaluation stopping criteria at {stopping_criteria_end} \n")
        print(f"Evaluation took {stopping_criteria_end - stopping_criteria_start} \n")
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
        "kappa_average_conservative_history": kappa_average_conservative_history,
        "kappa_average_middle_ground_history": kappa_average_middle_ground_history,
        "kappa_average_aggressive_history": kappa_average_aggressive_history,
        "delta_f_score_conservative_history": delta_f_score_conservative_history,
        "delta_f_score_middle_ground_history": delta_f_score_middle_ground_history,
        "delta_f_score_aggressive_history": delta_f_score_aggressive_history,
        "classification_change_conservative_history": classification_change_conservative_history,
        "classification_change_middle_ground_history": classification_change_middle_ground_history,
        "classification_change_aggressive_history": classification_change_aggressive_history,
        # "overall_uncertainty_conservative_history": overall_uncertainty_conservative_history,
        # "overall_uncertainty_middle_ground_history": overall_uncertainty_middle_ground_history,
        # "overall_uncertainty_aggressive_history": overall_uncertainty_aggressive_history,
        # "stopping_history": stopping_history,
        "samples_count": samples_count,
    }

    details_str = f"{query_strategy}_{base_strategy}_{num_queries}_queries_num_samples_{num_samples}_num_augmentations_{num_augmentations}"

    # with open((Path(__file__).parent / "../results" / saving_name).resolve(), "w") as f:
    # f.write(d_frame.to_json())

    return d, details_str
