"""The single interface every recommender implements."""
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Recommender(Protocol):
    """A recommender fits on training data and returns a ranked list of book ids.

    `query` is polymorphic per use case: a sequence of seen book ids (UC1),
    an anchor book id (UC4), or a natural-language string (UC3/UC5).
    """

    def fit(self, train_data: Any) -> "Recommender":
        ...

    def recommend(self, query: Any, k: int) -> list:
        ...
