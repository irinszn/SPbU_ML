import numpy as np
import pytest
from scipy.spatial.distance import euclidean

from src.homeworks.homework_2.kd_tree import KDTree


@pytest.fixture
def generate_data():
    dims = [2, 3, 5]
    train_dataset = [np.random.rand(150, dim) for dim in dims]
    test_dataset = [np.random.rand(30, dim) for dim in dims]
    return train_dataset, test_dataset


def brute_force(X, fixed_points, k):
    neighbours = []

    def brute_force_for_point(X, fixed_point, k):
        temp = []
        for point in X:
            temp.append((euclidean(point, fixed_point), point))
        return [np.array(point) for dist, point in sorted(temp, key=lambda t: t[0])[:k]]

    for point in fixed_points:
        neighbours.append(brute_force_for_point(X, point, k))
    return np.array(neighbours)


class TestKDTree:
    def test_kdtree_build(self):
        points = [(1, 2), (3, 4), (5, 6), (7, 8)]
        leaf_size = 2
        tree = KDTree(points, leaf_size=leaf_size)

        assert tree.root is not None
        assert len(tree.root.right.leaf_points) == 1 and len(tree.root.left.leaf_points) == 2

    def test_kdtree_query(self):
        points = [(1, 2), (3, 4), (5, 6), (7, 8)]
        input_points = [(2, 3), (4, 5)]

        leaf_size = 2
        tree = KDTree(points, leaf_size=leaf_size)
        neighbours = tree.query(input_points, k=2)

        expected_neighbours = np.array([[[1, 2], [3, 4]], [[3, 4], [5, 6]]])
        assert np.array_equal(neighbours, expected_neighbours)

    def test_kdtree_vs_brute_force(self, generate_data):
        train_dataset, test_dataset = generate_data

        leaf_size = 15

        for test, train in zip(train_dataset, test_dataset):
            tree = KDTree(train, leaf_size=leaf_size)
            neighbours_by_tree = tree.query(test, k=6)

            brute_neighbours = brute_force(train, test, k=6)
            assert np.array_equal(neighbours_by_tree, brute_neighbours)
