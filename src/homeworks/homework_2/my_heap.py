import heapq

import numpy as np


class MyHeap:
    def __init__(self, k: int):
        self.k = k
        self.heap: list[tuple] = []

    def add_element(self, dist_with_point: tuple):
        heapq.heappush(self.heap, dist_with_point)
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)

    def get_max_dist(self) -> float:
        if self.heap:
            return -self.heap[0][0]
        return float("inf")

    def get_sorted(self):
        return [np.array(point) for dist, point in sorted(self.heap, key=lambda t: -t[0])]
