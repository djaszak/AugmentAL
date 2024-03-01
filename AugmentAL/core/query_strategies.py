import nlpaug.augmenter.word as naw
import numpy as np
import time
import copy
from scipy.stats import entropy
from small_text.query_strategies import (
    ConfidenceBasedQueryStrategy,
    BreakingTies,
    QueryStrategy,
)


class AugmentedQueryStrategyBase(QueryStrategy):
    """This class serves as an abstract base for implementing QueryStrategies
    which expend other BaseStrategys by using an augmented dataset.
    """

    def __init__(self, augmented_indices: dict[int, list[int]] = {}) -> None:
        """Initialize the strategy by providing the augmented_indices for the dataset
            and the

        Args:
            augmented_indices (dict[int, list[int]], optional): A dictionary mapping samples to
                virtual samples generated based on this one. The key is always one
                id representing an element from the base set, the value to this
                will always be a list consisting of ids representing virtual samples
                generated based on the original one. Defaults to {}.
        """
        super().__init__()
        self.augmented_indices = augmented_indices
        self.flattened_augmented_values = sorted(
            {x for v in self.augmented_indices.values() for x in v}
        )

    def get_origin_augmented_index(self, aug_elem_index) -> int:
        # TODO: More robust by introducing try except for index error
        return [
            key
            for key in self.augmented_indices
            if (aug_elem_index in self.augmented_indices[key])
        ][0]


class AugmentedExtensionQueryStrategy(AugmentedQueryStrategyBase):
    """This strategy simply utilizes the extension of a dataset done by
    augmentation to give the base_strategy more samples to score.
    """

    def __init__(
        self, base_strategy: QueryStrategy, augmented_indices: dict[int, list[int]] = {}
    ) -> None:
        """Initialize by adding the base_strategy that will be used for calculating a score for
            the samples.

        Args:
            base_strategy (QueryStrategy): base_strategy (QueryStrategy): The strategy that will be used to calculate
                the scores. Can be any strategy provided by the small_text framework.
            augmented_indices (dict[int, list[int]], optional): See base class. Defaults to {}.
        """
        super().__init__(augmented_indices)
        self.base_strategy = base_strategy

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):
        # Firstly we need three lists to keep track of the indices
        # that we do not want to query again in while loop.
        original_indices_queried = np.array([], dtype=int)
        augmented_indices_queried = np.array([], dtype=int)
        already_queried = np.array([], dtype=int)

        while len(original_indices_queried) < n:
            # Two steps: 
            # 1. Query the base_strategy indices_unlabeled should
            # be trimmed of indices that we either queried already in the 
            # while loop or that we queried in the past.
            # 2. Add the indices that we queried to the already_queried list
            # get original indices, if we get augemented ones in base and
            # at the end get all augmented from the base ones. This
            # ensures that we can "trow away" the augmented ones after
            # the while loop if we want to.
            query = self.base_strategy.query(
                clf,
                dataset,
                np.setdiff1d(
                    np.setdiff1d(
                        np.setdiff1d(indices_unlabeled, already_queried),
                        augmented_indices_queried,
                    ),
                    original_indices_queried,
                ),
                indices_labeled,
                y,
                n - len(original_indices_queried),
            )

            already_queried = np.unique(np.concatenate((already_queried, query)))
            original_indices_queried = np.unique(
                np.concatenate(
                    (original_indices_queried,
                    [
                        (
                            int(self.get_origin_augmented_index(elem))
                            if elem in self.flattened_augmented_values
                            else int(elem)
                        )
                        for elem in query
                    ]),
                )
            )
            augmented_indices_queried = np.unique( 
                np.concatenate(
                    (augmented_indices_queried,
                    [
                        int(x)
                        for xs in [
                            self.augmented_indices[x] for x in original_indices_queried
                        ]
                        for x in xs
                    ]),
                )
            )

        original_indices_queried = original_indices_queried[:n]
        return np.concatenate((original_indices_queried, augmented_indices_queried))


# AugmentedLeastConfidenceQueryStrategy
class AugmentedEntropyQueryStrategy(
    AugmentedQueryStrategyBase, ConfidenceBasedQueryStrategy
):
    """Selects instances which have a big entropy among themselves and
    artifially created instances.
    """

    def get_confidence(self, clf, dataset, indices_unlabeled, indices_labeled, y):
        # Use the best confidence from the classifier
        proba = clf.predict_proba(dataset).max(axis=1)

        for i in self.augmented_indices:
            proba[i] = np.mean(
                np.concatenate(
                    ([proba[i]], [proba[x] for x in self.augmented_indices[i]])
                )
            )
            for x in self.augmented_indices[i]:
                proba[x] = proba[i]

        return proba

    def __str__(self):
        return "AugmentedEntropyQueryStrategy"
