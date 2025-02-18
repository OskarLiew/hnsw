from bisect import insort
from collections import defaultdict
from heapq import heappop, heappush
import math
import random

from common import cosine_similarity, random_vector


class HNSW:
    def __init__(self, m_max: int, ef_construction: int, n_layers: int) -> None:
        self.m_max = m_max
        self.m_max_0 = m_max
        self.ef_construction = ef_construction
        self.n_layers = n_layers

        # id => vector
        self.vectors: dict[int, list[float]] = {}

        # id => layer => [(dist, child_id)]
        self.neighbours: dict[int, dict[int, list[tuple[float, int]]]] = defaultdict(
            dict
        )

    @property
    def m_total(self) -> int:
        return len(self.vectors)

    def insert(self, vector: list[float]) -> None:
        insert_idx = self.m_total
        self.vectors[insert_idx] = vector

        # If first node, insert to all layers
        if len(self.vectors) == 1:
            insert_layer = self.n_layers - 1
            for i_layer in range(self.n_layers - 1, -1, -1):
                self.neighbours[i_layer][insert_idx] = []
            return

        # Select layer exponentially distributed
        insert_layer = math.floor(
            -math.log(random.random()) * 1 / math.log(self.m_total)
        )
        insert_layer = min(insert_layer, self.n_layers - 1)

        # Find start point in insert layer
        start_idx = 0
        for i_layer in range(self.n_layers - 1, insert_layer, -1):
            neighbours = self._search_layer(start_idx, vector, i_layer, ef=1)
            start_idx = neighbours[0][1]

        # Go through all layers from insert layer and down
        for i_layer in range(insert_layer, -1, -1):
            # Find the nearest neighbours of the vector
            neighbours = self._search_layer(
                start_idx, vector, i_layer, ef=self.ef_construction
            )

            # Set neighbours. Use m_max_0 if bottom layer
            m_max = self.m_max if i_layer else self.m_max_0
            self.neighbours[i_layer][insert_idx] = neighbours[:m_max]

            # Update neighbour edges
            for n_sim, n_id in neighbours:
                # Could use more advanced select here
                insort(self.neighbours[i_layer][n_id], (n_sim, insert_idx))
                self.neighbours[i_layer][n_id] = self.neighbours[i_layer][n_id][:m_max]

            start_idx = neighbours[0][1]  # OR maybe all?

    def _search_layer(
        self, start_idx: int, vector: list[float], layer: int, ef: int
    ) -> list[tuple[float, int]]:
        sim = cosine_similarity(vector, self.vectors[start_idx])

        visited = {start_idx}
        candidates = [(sim, start_idx)]
        neighbours = [(sim, start_idx)]

        while candidates:
            candidate = heappop(candidates)
            furthest_neigh = neighbours[-1]

            if candidate[0] > furthest_neigh[0]:
                break

            for c_neigh in self.neighbours[layer][candidate[1]]:
                if c_neigh in visited:
                    continue

                visited.add(c_neigh[1])
                furthest_neigh = neighbours[-1]
                neigh_sim = cosine_similarity(vector, self.vectors[c_neigh[1]])
                if neigh_sim < neighbours[0][0]:
                    heappush(candidates, (neigh_sim, c_neigh[1]))
                    insort(neighbours, (neigh_sim, c_neigh[1]))
                    if len(neighbours) > ef:
                        neighbours.pop()

        return neighbours

    def search(self, vector: list[float], ef: int) -> list[tuple[float, int]]:
        start_idx = 0
        neighbours = []
        for i_layer in range(self.n_layers - 1, -1, -1):
            neighbours = self._search_layer(start_idx, vector, i_layer, ef)
            start_idx = neighbours[0][-1]
        return neighbours


if __name__ == "__main__":
    index = HNSW(3, 8, 3)
    for v in (random_vector(8) for _ in range(100)):
        index.insert(v)

    # __import__("pprint").pprint(index.neighbours)
    print(index.search(random_vector(8), 8))
