import datasets
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
)
from constants import TransformerModels
from augment import create_augmented_dataset


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


def warm_start_active_learner(active_learner, y_train):
    """Warm start the sample pool"""

    indices_initial = random_initialization_balanced(y_train, n_samples=20)
    active_learner.initialize_data(indices_initial, y_train[indices_initial])

    return indices_initial


def create_active_learner(
    train_set: TransformersDataset,
    num_classes: int,
    query_strategy: QueryStrategy = PredictionEntropy,
) -> set[PoolBasedActiveLearner, int]:
    """Load transformer, build clf_factory based on it and return a PoolBasedActiveLearner.

    Args:
        train_set (TransformersDataset): A training set
        num_classes (int): Number of labels

    Returns:
        set[PoolBasedActiveLearner, int]: The active learner and the indices of pre_labeled data after warm start
    """
    transformer_model = TransformerModelArguments(TransformerModels.BERT_TINY.value)
    clf_factory = TransformerBasedClassificationFactory(
        transformer_model,
        num_classes,
        kwargs=dict({"mini_batch_size": 32, "class_weight": "balanced"}),
    )

    active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train_set)
    indices_labeled = warm_start_active_learner(active_learner, train_set.y)

    return active_learner, indices_labeled


def evaluate(active_learner, train, test):
    y_pred = active_learner.classifier.predict(train)
    y_pred_test = active_learner.classifier.predict(test)

    test_acc = accuracy_score(y_pred_test, test.y)

    print("Train accuracy: {:.2f}".format(accuracy_score(y_pred, train.y)))
    print("Test accuracy: {:.2f}".format(test_acc))

    return test_acc
