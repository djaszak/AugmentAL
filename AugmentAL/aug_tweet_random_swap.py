from core.constants import AugmentationMethods, Datasets
from script import create_raw_set

print(f"starting augmentation of {Datasets.TWEET.value} with {AugmentationMethods.RANDOM_SWAP.value}")
raw_test, _, _ =create_raw_set(Datasets.TWEET.value, AugmentationMethods.RANDOM_SWAP.value)
print("finished augmentation")
print(raw_test)

