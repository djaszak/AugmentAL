from script import run_script
from core.constants import Datasets

run_script(
    query_strategies=[
        "RandomSampling",
    ],
    dataset=Datasets.IMDB.value,
)
