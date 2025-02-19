import math
import random
import time
import bisect

from common import random_vector, cosine_distance

DIM = 32


class Node:
    def __init__(
        self,
        vector: list[float],
        neighbours: list[tuple["Node", float]] | None = None,
        parent: "Node | None" = None,
    ) -> None:
        self.vector = vector
        self.neighbours = neighbours or []
        self.parent = parent

    def get_neighbours(self):
        return [n[0] for n in self.neighbours]


class HNSWEF:
    def __init__(
        self,
        n_layers: int,
        n_edges: int = 8,
        p_layer: float | None = None,
        ef_construct: int = 16,
    ) -> None:
        self.layers: list[list[Node]] = [[] for _ in range(n_layers)]
        self.n_edges = (2 * n_edges, n_edges)
        self.p_layer = p_layer if p_layer is not None else 1 / math.log(n_edges)
        self.ef_construct = ef_construct

    def add_node(self, vector: list[float]) -> None:
        parent = None
        for i in range(len(self.layers)):
            parent = self._add_to_layer(vector, i, parent)

            # Add to next layer if it's empty, otherwise with probability p_l
            if (
                i + 1 < len(self.layers) and self.layers[i + 1]
            ) and random.random() > self.p_layer:
                break

    def _add_to_layer(
        self, vector: list[float], layer_idx: int, parent: Node | None
    ) -> Node:
        node_sims = self.layer_search(vector, layer_idx, ef=self.ef_construct)

        k_layer = round(
            self.n_edges[1]
            + (self.n_edges[0] - self.n_edges[1])
            * (len(self.layers) - layer_idx)
            / len(self.layers)
        )
        new_node = Node(vector, neighbours=node_sims[:k_layer], parent=parent)
        for n, sim in node_sims[:k_layer]:
            n.neighbours.append((new_node, sim))
            n.neighbours.sort(key=lambda x: x[1])
            n.neighbours = n.neighbours[-k_layer:]

        self.layers[layer_idx].append(new_node)
        return new_node

    def layer_search(
        self,
        search_vector: list[float],
        layer_idx: int,
        start_node: Node | None = None,
        ef: int = 16,
    ) -> list[tuple[Node, float]]:
        nodes = self.layers[layer_idx]
        if not nodes:
            return []

        if start_node is None:
            start_node = random.choice(nodes)

        # Use ef = 1 for upper layers
        ef = ef if layer_idx == 0 else max(4, ef // 2)

        res = nsw_search(search_vector, start_node, ef=ef)
        return res

    def search(
        self, search_vector: list[float], ef: int = 8
    ) -> list[tuple[Node, float]]:
        start_node = None
        node_sims = []

        for layer_idx in reversed(range(len(self.layers))):
            node_sims = self.layer_search(search_vector, layer_idx, start_node, ef=ef)

            best_node = node_sims[-1][0]
            if not best_node.parent:
                break

            start_node = best_node.parent

        return node_sims


def nsw_search(
    search_vector: list[float], start_node: Node, ef: int = 8
) -> list[tuple[Node, float]]:
    node = start_node

    out = [(node, cosine_distance(search_vector, node.vector))]
    visited = {node}
    candidates = out.copy()

    while candidates:
        candidate_node, _ = candidates.pop(-1)

        for neigh_node in candidate_node.get_neighbours():
            if neigh_node in visited:
                continue

            visited.add(neigh_node)
            neigh_sim = cosine_distance(search_vector, neigh_node.vector)
            if neigh_sim > out[-1][1] or len(out) < ef:
                bisect.insort(candidates, (neigh_node, neigh_sim), key=lambda x: x[1])
                bisect.insort(out, (neigh_node, neigh_sim), key=lambda x: x[1])

                if len(out) > ef:
                    out.pop(0)

    return out


def main():
    print("Starting")
    vectors = [random_vector(DIM) for _ in range(2000)]

    print("Indexing")
    t0 = time.time()
    # Index
    index = HNSWEF(n_layers=4, n_edges=5, ef_construct=32)
    for v in vectors:
        index.add_node(v)

    index_time = time.time() - t0
    print(f"Indexing took: {index_time:.5f} seconds")

    # Search
    search_vector = random_vector(DIM)

    print("Searching")

    # NSW
    t0 = time.time()
    hnsw_sims = index.search(search_vector, ef=16)
    hnsw_time = time.time() - t0

    best_node_sim = max(hnsw_sims, key=lambda x: x[1])[1]
    print("HNSW", best_node_sim, hnsw_time)

    # Naive
    t0 = time.time()
    naive_best = max(cosine_distance(search_vector, v) for v in vectors)
    naive_time = time.time() - t0

    print("Naive", naive_best, naive_time)


if __name__ == "__main__":
    main()
