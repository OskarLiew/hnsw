from bisect import insort
from heapq import heapify, heappop, heappush
import math
import random
import time

from common import cosine_similarity, random_vector

DIM = 32


class HNSW:
    def __init__(
        self,
        m_max: int,
        ef_construction: int,
        n_layers: int,
        m_l: float | None = None,
        prune_connections: bool = False,
        extend_candidates: bool = False,
    ) -> None:
        self.m_max = m_max
        self.m_max_0 = m_max * 2
        self.ef_construction = ef_construction
        self.n_layers = n_layers
        self.m_l = m_l
        self.prune_connections = prune_connections
        self.extend_candidates = extend_candidates

        # id => vector
        self.vectors: dict[int, list[float]] = {}

        # id => layer => [(dist, child_id)]
        self.neighbours: dict[int, dict[int, list[tuple[float, int]]]] = {
            i: {} for i in range(n_layers)
        }

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
        m_l = self.m_l if self.m_l else 1 / math.log(self.m_total)
        insert_layer = math.floor(-math.log(random.random()) * m_l)
        insert_layer = min(insert_layer, self.n_layers - 1)

        # Find start point in insert layer
        start_indices = [0]
        for i_layer in range(self.n_layers - 1, insert_layer, -1):
            neighbours = self._search_layer(start_indices, vector, i_layer, ef=1)
            start_indices = [neighbours[0][1]]

        # Go through all layers from insert layer and down
        for i_layer in range(insert_layer, -1, -1):
            # Find the nearest neighbours of the vector
            neighbours = self._search_layer(
                start_indices, vector, i_layer, ef=self.ef_construction
            )

            # Set neighbours. Use m_max_0 if bottom layer
            m_max = self.m_max if i_layer else self.m_max_0
            self.neighbours[i_layer][insert_idx] = neighbours[:m_max]

            # Update neighbour edges
            for n_sim, n_id in neighbours:
                candidates = self.neighbours[i_layer][n_id].copy()
                insort(candidates, (n_sim, insert_idx))
                self.neighbours[i_layer][n_id] = self._select_neighbours(
                    i_layer,
                    candidates,
                    m=m_max,
                    extend_candidates=self.extend_candidates,
                    prune_connections=self.prune_connections,
                )

            start_indices = [n[1] for n in neighbours]

    def _search_layer(
        self, start_indices: list[int], vector: list[float], layer: int, ef: int
    ) -> list[tuple[float, int]]:
        visited = set(start_indices)
        neighbours = [
            (cosine_similarity(vector, self.vectors[i]), i) for i in start_indices
        ]

        candidates = neighbours.copy()
        heapify(candidates)

        while candidates:
            candidate = heappop(candidates)
            furthest_neigh = neighbours[-1]

            if candidate[0] > furthest_neigh[0]:
                break

            for c_neigh in self.neighbours[layer][candidate[1]]:
                if c_neigh[1] in visited:
                    continue

                visited.add(c_neigh[1])
                furthest_neigh = neighbours[-1]
                neigh_sim = cosine_similarity(vector, self.vectors[c_neigh[1]])
                if neigh_sim < neighbours[0][0] or len(neighbours) < ef:
                    heappush(candidates, (neigh_sim, c_neigh[1]))
                    insort(neighbours, (neigh_sim, c_neigh[1]))
                    if len(neighbours) > ef:
                        neighbours.pop()

        return neighbours

    def search(self, vector: list[float], ef: int) -> list[tuple[float, int]]:
        start_indices = [0]
        neighbours = []
        for i_layer in range(self.n_layers - 1, -1, -1):
            neighbours = self._search_layer(start_indices, vector, i_layer, ef)
            start_indices = [neighbours[0][1]]
        return neighbours

    def _select_neighbours(
        self,
        layer: int,
        candidates: list[tuple[float, int]],
        m: int,
        extend_candidates: bool,
        prune_connections: bool,
    ) -> list[tuple[float, int]]:
        out = []
        neighbours = candidates.copy()

        # Extend search to include neighbours of neighbours
        if extend_candidates:
            for candidate in candidates:
                for candidate_adj in self.neighbours[layer][candidate[1]]:
                    if candidate_adj not in neighbours:
                        insort(neighbours, candidate_adj)

        if not prune_connections:
            return neighbours[:m]

        # Add all neighbours that are sufficiently close
        while neighbours and len(out) < m:
            candidate = neighbours.pop(0)
            if not out or candidate[0] < out[0][0]:
                insort(out, candidate)

        return out


def main():
    print("Starting")
    vectors = [random_vector(DIM) for _ in range(2000)]
    search_vector = random_vector(DIM)

    print("Indexing")
    t0 = time.time()

    ### HNSW
    # Index
    index = HNSW(
        n_layers=4,
        m_max=24,
        m_l=None,
        ef_construction=32,
        prune_connections=False,
        extend_candidates=False,
    )
    for v in vectors:
        index.insert(v)

    for l, d in index.neighbours.items():
        print(l, len(d))

    index_time = time.time() - t0
    print(f"Indexing took: {index_time:.5f} seconds")

    # Search
    print("Searching")
    t0 = time.time()
    hnsw_sims = index.search(search_vector, ef=16)
    hnsw_time = time.time() - t0
    print("HNSW", hnsw_sims[0][0], hnsw_time)

    # NSW
    index.neighbours = {0: index.neighbours[0]}
    index.n_layers = 1
    print("Searching")
    t0 = time.time()
    hnsw_sims = index.search(search_vector, ef=16)
    hnsw_time = time.time() - t0
    print("NSW", hnsw_sims[0][0], hnsw_time)

    # Naive
    t0 = time.time()
    naive_best = min(cosine_similarity(search_vector, v) for v in vectors)
    naive_time = time.time() - t0

    print("Naive", naive_best, naive_time)


if __name__ == "__main__":
    main()
