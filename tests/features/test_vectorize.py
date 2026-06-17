import scipy.sparse as sp

from book_recsys.features.vectorize import bow_matrix, tfidf_matrix

DOCS = [
    "Title: Dragon\nPlot: a young mage rises against darkness",
    "Title: Wizard\nPlot: a young mage battles darkness",
    "Title: Romance\nPlot: two lovers meet in spring",
]


def test_tfidf_matrix_shape_and_type():
    matrix, vec = tfidf_matrix(DOCS, max_features=50)
    assert sp.issparse(matrix)
    assert matrix.shape[0] == 3
    assert matrix.shape[1] == len(vec.get_feature_names_out())


def test_bow_matrix_counts_are_integers():
    matrix, _ = bow_matrix(DOCS, max_features=50)
    assert sp.issparse(matrix)
    assert matrix.shape[0] == 3
    assert (matrix.data == matrix.data.astype(int)).all()


def test_max_features_caps_vocabulary():
    matrix, _ = tfidf_matrix(DOCS, max_features=2)
    assert matrix.shape[1] == 2
