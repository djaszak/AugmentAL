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

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):
        query = self.base_strategy.query(
            clf, dataset, indices_unlabeled, indices_labeled, y, n
        )

        # Switch out elements found by augmentation with origin element
        for elem in query:
            if elem in self.augmented_indices.values():
                query[
                    query == elem
                ] = self.get_origin_augmented_index(elem)

        return query


class AugmentedBreakingTiesQueryStrategy(BreakingTies):
    def __init__(self, lower_is_better=False, augmented_indices={}, base_strategy=None):
        self.lower_is_better = lower_is_better
        self.scores_ = None
        self.augmented_indices = augmented_indices

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):
        self._validate_query_input(indices_unlabeled, n)

        confidence = self.base_strategy.score(
            clf, dataset, indices_unlabeled, indices_labeled, y
        )

        if len(indices_unlabeled) == n:
            return np.array(indices_unlabeled)

        indices_partitioned = np.argpartition(confidence[indices_unlabeled], n)[:n]

        # Switch out elements found by augmentation with origin element
        for elem in indices_partitioned:
            if elem in self.augmented_indices.values():
                indices_partitioned[
                    indices_partitioned == elem
                ] = self.get_origin_augmented_index(elem)

        return np.array([indices_unlabeled[i] for i in indices_partitioned])


class AugmentedEntropyQueryStrategy(ConfidenceBasedQueryStrategy):
    """Selects instances which have a big entropy among themselves and
    artifially created instances.
    """

    def __init__(self, augmented_indices=[], lower_is_better=False):
        super().__init__(lower_is_better=lower_is_better)
        self.augmented_indices = augmented_indices

    def get_confidence(self, clf, dataset, _indices_unlabeled, _indices_labeled, _y):
        # TODO: Here we have to add _indices_augmented: dict[origin_id: list[augmented_id]]
        proba = clf.predict_proba(dataset)
        print(proba)
        return np.apply_along_axis(lambda x: self._best_versus_second_best(x), 1, proba)

    # def _augment_data(dataset):
    #     aug = naw.SynonymAug(aug_src="wordnet")
    #     augmented_text = aug.augment(text)
    #     print("Original:")
    #     print(text)
    #     print("Augmented Text:")
    #     print(augmented_text)

    @staticmethod
    def _best_versus_second_best(proba):
        ind = np.argsort(proba)
        return proba[ind[-1]] - proba[ind[-2]]

    def __str__(self):
        return "BreakingTies()"


class AdditionalAugmentedDataQueryStrategy:
    pass
