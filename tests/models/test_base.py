from book_recsys.models.base import Recommender


class _Dummy:

    def fit(self, train_data):
        return self

    def recommend(self, query, k):
        return ["b0"][:k]


def test_dummy_satisfies_protocol():
    assert isinstance(_Dummy(), Recommender)


def test_non_conforming_object_fails_protocol():
    assert not isinstance(object(), Recommender)
