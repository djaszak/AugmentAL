from core.constants import AugmentationMethods, Datasets
from script import create_raw_set

print(f"starting augmentation of {Datasets.AG_NEWS.value} with {AugmentationMethods.BERT_SUBSTITUTE.value}")
raw_test, _, _ =create_raw_set(Datasets.AG_NEWS.value, AugmentationMethods.BERT_SUBSTITUTE.value)
print("finished augmentation")
print(raw_test)

