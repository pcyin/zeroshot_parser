import math
from typing import List, Any, Iterator


def get_batches(
    dataset: List[Any],
    batch_size: int
) -> Iterator[List[Any]]:
    batch_num = math.ceil(len(dataset) / batch_size)
    for i in range(batch_num):
        examples = dataset[i * batch_size: (i + 1) * batch_size]

        yield examples
