from core.constants import AugmentationMethods, Datasets
from script import create_raw_set

print(
    f"starting augmentation of {Datasets.AG_NEWS.value} with {AugmentationMethods.BACK_TRANSLATION.value}"
)
raw_test, _, _ = create_raw_set(
    Datasets.AG_NEWS.value, AugmentationMethods.BACK_TRANSLATION.value
)
print("finished augmentation")
print(raw_test)

# from core.constants import AugmentationMethods, Datasets
# from script import create_raw_set
# import multiprocessing
# import torch.multiprocessing as mp
# if __name__ == '__main__':
#     mp.set_start_method('spawn')
#     print(f"starting augmentation of {Datasets.AG_NEWS.value} with {AugmentationMethods.BACK_TRANSLATION.value}")
#     raw_test, _, _ =create_raw_set(Datasets.AG_NEWS.value, AugmentationMethods.BACK_TRANSLATION.value)
#     print("finished augmentation")
#     print(raw_test)
