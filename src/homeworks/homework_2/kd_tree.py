from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import euclidean

from src.homeworks.homework_2.my_heap import MyHeap


@dataclass
class Node:
    point: Optional[NDArray] = None
    axis: Optional[int] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    leaf_points: Optional[NDArray] = None


class KDTree:
    def __init__(self, points: list[tuple[float, ...]], metric: Callable = euclidean, leaf_size: int = 10) -> None:
        if not isinstance(points, Iterable) or len(points) == 0:
            raise ValueError("list of points can't be null.")

        if not isinstance(leaf_size, int) or leaf_size <= 0:
            raise ValueError("leaf_size must be positive and integer.")

        dim = len(points[0])
        for point in points:
            if not isinstance(point, Iterable) or len(point) != dim:
                raise ValueError("all points must be tuples of the same dimension.")

        if not callable(metric):
            raise ValueError("metric must be function.")

        self.leaf_size = leaf_size
        self.root = self.build_tree(np.array(points))
        self.metric = metric

    def build_tree(self, points: NDArray) -> Node:
        if len(points) <= self.leaf_size:
            return Node(leaf_points=points)

        axis = np.argmax(np.var(points, axis=0))

        points = points[points[:, axis].argsort()]
        median_index = len(points) // 2

        return Node(
            point=points[median_index],
            axis=int(axis),
            left=self.build_tree(points[:median_index]),
            right=self.build_tree(points[median_index + 1 :]),
        )

    def query(self, points: list[tuple[float, ...]], k: int = 1) -> NDArray:
        if not isinstance(points, Iterable) or len(points) == 0:
            raise ValueError("list of points can't be null.")

        dim = len(points[0])
        for point in points:
            if not isinstance(point, Iterable) or len(point) != dim:
                raise ValueError("all points must be tuples of the same dimension.")

        if not isinstance(k, int) or k <= 0:
            raise ValueError("leaf_size must be positive and integer.")

        query_points = np.array(points)
        neighbours = [self.query_point(point, k) for point in query_points]

        return np.array(neighbours)

    def query_point(self, fixed_point: NDArray, k: int) -> list[NDArray]:
        heap = MyHeap(k)

        def search_rec(node: Optional[Node]) -> None:
            if node is None:
                return

            if node.leaf_points is not None:
                for point in node.leaf_points:
                    dist = self.metric(point, fixed_point)
                    heap.add_element((-dist, point.tolist()))
                return

            dist = self.metric(node.point, fixed_point)

            if node.point is not None and dist < heap.get_max_dist():
                heap.add_element((-dist, node.point.tolist()))

            if node.point is not None and fixed_point[node.axis] < node.point[node.axis]:
                search_rec(node.left)
                if (
                    node.point is not None
                    and np.abs(node.point[node.axis] - fixed_point[node.axis]) < heap.get_max_dist()
                ):
                    search_rec(node.right)
            else:
                search_rec(node.right)
                if (
                    node.point is not None
                    and np.abs(node.point[node.axis] - fixed_point[node.axis]) < heap.get_max_dist()
                ):
                    search_rec(node.left)

        search_rec(self.root)
        return heap.get_sorted()
