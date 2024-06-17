from core.constants import AugmentationMethods, Datasets
from script import create_raw_set

print(
    f"starting augmentation of {Datasets.TWEET.value} with {AugmentationMethods.BACK_TRANSLATION.value}"
)
raw_test, _, _ = create_raw_set(
    Datasets.TWEET.value, AugmentationMethods.BACK_TRANSLATION.value
)
print("finished augmentation")
print(raw_test)
