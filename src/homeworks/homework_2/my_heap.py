import heapq

import numpy as np
from numpy.typing import NDArray


class MyHeap:
    def __init__(self, k: int) -> None:
        self.k = k
        self.heap: list[tuple] = []

    def add_element(self, dist_with_point: tuple) -> None:
        heapq.heappush(self.heap, dist_with_point)
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)

    def get_max_dist(self) -> float:
        if self.heap:
            return -self.heap[0][0]
        return float("inf")

    def get_sorted(self) -> list[NDArray]:
        return [np.array(point) for dist, point in sorted(self.heap, key=lambda t: -t[0])]
