import nlpaug.augmenter.word as naw
import numpy as np
from small_text.query_strategies import (
    ConfidenceBasedQueryStrategy,
    BreakingTies,
    QueryStrategy,
)


class AugmentedQueryStrategyBase(QueryStrategy):
    """This class serves as an abstract base for implementing QueryStrategies
    which expend other BaseStrategys by using an augmented dataset.
    """

    def __init__(
        self, base_strategy: QueryStrategy, augmented_indices: dict[int, list[int]] = {}
    ) -> None:
        """Initialize the strategy by providing the augmented_indices for the dataset
            and the base_strategy that will be used for calculating a score for
            the samples.

        Args:
            base_strategy (QueryStrategy): The strategy that will be used to calculate
                the scores. Can be any strategy provided by the small_text framework.
            augmented_indices (dict[int, list[int]], optional): A dictionary mapping samples to
                virtual samples generated based on this one. The key is always one
                id representing an element from the base set, the value to this
                will always be a list consisting of ids representing virtual samples
                generated based on the original one. Defaults to {}.
        """
        super().__init__()
        self.base_strategy = base_strategy
        self.augmented_indices = augmented_indices

    def get_origin_augmented_index(self, aug_elem_index):
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

    def query(
        self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10
    ) -> np.ndarray[int]:
        query = self.base_strategy.query(
            clf, dataset, indices_unlabeled, indices_labeled, y, n
        )
        # Flatten the values. Augmented_indices should be unique mappings.
        flattened_augmented_values = sorted(
            {x for v in self.augmented_indices.values() for x in v}
        )
        original_queried_indices = [
            (
                self.get_origin_augmented_index(elem)
                if elem in flattened_augmented_values
                else elem
            )
            for elem in query
        ]
        augmented_indices_queried = [
            x
            for xs in [self.augmented_indices[x] for x in original_queried_indices]
            for x in xs
        ]
        return np.unique(np.array(original_queried_indices + augmented_indices_queried))


class AugmentedEntropyQueryStrategy(ConfidenceBasedQueryStrategy):
    """Selects instances which have a big entropy among themselves and
    artifially created instances.
    """
    pass
