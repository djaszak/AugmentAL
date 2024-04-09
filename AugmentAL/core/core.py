import datasets
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from matplotlib import rcParams
from transformers import AutoTokenizer
from small_text import (
    PoolBasedActiveLearner,
    PredictionEntropy,
    TransformerBasedClassificationFactory,
    TransformersDataset,
    TransformerModelArguments,
    random_initialization_balanced,
    QueryStrategy,
    Classifier,
)
from core.constants import TransformerModels


# CONSTANTS
SEED = 2022

# SETUP
datasets.logging.set_verbosity_error()
rcParams.update({"xtick.labelsize": 14, "ytick.labelsize": 14, "axes.labelsize": 16})
np.random.seed(SEED)

# LOAD SET


def create_small_text_dataset(
    dataset: datasets.Dataset,
) -> set[TransformersDataset, TransformersDataset, int]:
    """Get specified hf dataset and transform to small text train and test set

    Args:
        dataset (datasets.Dataset): An hf dataset partition.

    Returns:
        tuple[TransformersDataset, TransformersDataset, int]: A train and a test set and the num_labels
    """
    tokenizer = AutoTokenizer.from_pretrained(TransformerModels.BERT_TINY.value)

    num_classes = dataset.features["label"].num_classes

    # TRANSFORM INTO SMALL TEXT
    target_labels = np.arange(num_classes)

    return TransformersDataset.from_arrays(
        dataset["text"],
        dataset["label"],
        tokenizer,
        max_length=60,
        target_labels=target_labels,
    )


def warm_start_active_learner(
    active_learner, y_train, training_indices: np.ndarray = None
):
    """Warm start the sample pool"""

    training_indices = (
        y_train[training_indices] if training_indices is not None else y_train
    )
    indices_initial = random_initialization_balanced(y_train, n_samples=20)
    active_learner.initialize_data(indices_initial, y_train[indices_initial])

    return indices_initial


def create_active_learner(
    train_set: TransformersDataset,
    num_classes: int,
    query_strategy: QueryStrategy = PredictionEntropy,
    training_indices: np.ndarray = None,
) -> set[PoolBasedActiveLearner, int]:
    """Load transformer, build clf_factory based on it and return a PoolBasedActiveLearner.

    Args:
        train_set (TransformersDataset): A training set
        num_classes (int): Number of labels

    Returns:
        set[PoolBasedActiveLearner, int]: The active learner and the indices of pre_labeled data after warm start
    """
    transformer_model = TransformerModelArguments(TransformerModels.BERT.value)
    
    clf_factory = TransformerBasedClassificationFactory(
        transformer_model,
        num_classes,
        kwargs={
                    'device': 'cuda', 
                    'mini_batch_size': 32,
                    'class_weight': 'balanced'
                },
    )

    active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train_set)
    indices_labeled = warm_start_active_learner(
        active_learner, train_set.y, training_indices
    )

    return active_learner, indices_labeled


def evaluate(active_learner, train, test):
    y_pred = active_learner.classifier.predict(train)
    y_pred_test = active_learner.classifier.predict(test)
    
    # Notice: We observe the train accuracy now.
    train_acc = accuracy_score(y_pred, train.y)

    print('Train accuracy: {:.2f}'.format(train_acc))
    print('Test accuracy: {:.2f}'.format(accuracy_score(y_pred_test, test.y)))
    
    return train_acc


def fill_query_with_augmented_search_room(
    query_strategy: QueryStrategy,
    clf: Classifier,
    dataset: TransformersDataset,
    indices_unlabeled: np.ndarray,
    indices_labeled: np.ndarray,
    y: np.ndarray,
    n: int,
) -> set[np.ndarray, np.ndarray]:
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
        query = query_strategy.base_strategy.query(
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
                (
                    original_indices_queried,
                    [
                        (
                            int(query_strategy.get_origin_augmented_index(elem))
                            if elem in query_strategy.flattened_augmented_values
                            else int(elem)
                        )
                        for elem in query
                    ],
                ),
            )
        )
        augmented_indices_queried = np.unique(
            np.concatenate(
                (
                    augmented_indices_queried,
                    [
                        int(x)
                        for xs in [
                            query_strategy.augmented_indices[x]
                            for x in original_indices_queried
                        ]
                        for x in xs
                    ],
                ),
            )
        )
    original_indices_queried = original_indices_queried[:n]
    return original_indices_queried, augmented_indices_queried
