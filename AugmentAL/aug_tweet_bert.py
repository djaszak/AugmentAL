from core.constants import AugmentationMethods, Datasets
from script import create_raw_set

print(
    f"starting augmentation of {Datasets.TWEET.value} with {AugmentationMethods.BERT_SUBSTITUTE.value}"
)
raw_test, _, _ = create_raw_set(
    Datasets.TWEET.value, AugmentationMethods.BERT_SUBSTITUTE.value
)
print("finished augmentation")
print(raw_test)
