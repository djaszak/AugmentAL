from core.augment import create_augmented_dataset
from core.constants import AugmentationMethods, Datasets
from datasets import load_dataset

# dataset = load_dataset(Datasets.ROTTEN.value)

# raw_test = dataset["test"]
# raw_train = dataset["train"].select([0, 10, 20, 30, 40, 50])
# augmented_indices = {}

# raw_train, augmented_indices = create_augmented_dataset(
#     raw_train, AugmentationMethods.BART_SUBSTITUTE.value, n=2
# )

# for key, value in augmented_indices.items():
#     print(raw_train[key], "\n")
#     for x in value:
#         print(raw_train[x], "\n") 

text = 'The quick brown fox jumps over the lazy dog.'

# print(f"Bart Substitute: {AugmentationMethods.BART_SUBSTITUTE.value.augment(text)}")
# print(f"Bert Substitute: {AugmentationMethods.BERT_SUBSTITUTE.value.augment(text)}")
# print(f"Abstractive Summarization: {AugmentationMethods.ABSTRACTIVE_SUMMARIZATION.value.augment(text)}")
# print(f"Random Swap: {AugmentationMethods.RANDOM_SWAP.value.augment(text)}")
# print(f"GPT2: {AugmentationMethods.GENERATIVE_GPT2.value.augment(text)}")
# print(f"DISTILLGPT2: {AugmentationMethods.GENERATIVE_DISTILGPT2.value.augment(text)}")
# print(f"Synonym with WordNet: {AugmentationMethods.SYNONYM.value.augment(text)}")
# print(f"Backtranslation: {AugmentationMethods.BACK_TRANSLATION.value.augment(text)}")

# GENERATIVE_GPT2 = nas.ContextualWordEmbsForSentenceAug(
#         model_path="gpt2", device="cuda"
# )
# GENERATIVE_DISTILGPT2 = nas.ContextualWordEmbsForSentenceAug(
#     model_path="distilgpt2", device="cuda"
# )

for x in range(5):
    print(f"GPT2: {AugmentationMethods.GENERATIVE_GPT2.value.augment(text)}")
    print(f"DISTILLGPT2: {AugmentationMethods.GENERATIVE_DISTILGPT2.value.augment(text)}") 
