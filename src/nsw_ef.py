import random
import time

from common import random_vector, cosine_similarity

DIM = 32


class Node:
    def __init__(
        self, vector: list[float], neighbours: list[tuple["Node", float]] | None = None
    ) -> None:
        self.vector = vector
        self.neighbours = neighbours or []

    def get_neighbours(self):
        return [n[0] for n in self.neighbours]


class NSW:
    def __init__(self, n_edges: int = 5, ef_construct: int = 16) -> None:
        self.nodes = []
        self.n_edges = n_edges
        self.ef_construct = ef_construct

    def add_node(self, vector: list[float]) -> None:
        node_sims = self.search(vector, ef=self.ef_construct)
        new_node = Node(vector, neighbours=node_sims[-self.n_edges :])
        for n, sim in node_sims[-self.n_edges :]:
            n.neighbours = sorted(n.neighbours + [(new_node, sim)], key=lambda x: x[1])[
                -self.n_edges :
            ]

        self.nodes.append(new_node)

    def search(
        self,
        search_vector: list[float],
        ef: int = 1,
    ) -> list[tuple[Node, float]]:
        if not self.nodes:
            return []
        return nsw_search(search_vector, self.nodes, ef=ef)


def nsw_search(
    search_vector: list[float], nodes: list[Node], ef: int
) -> list[tuple[Node, float]]:
    node = random.choice(nodes)

    out = [(node, cosine_similarity(search_vector, node.vector))]
    visited = {node}
    candidates = out.copy()

    while candidates:
        candidate_node, _ = candidates.pop(-1)

        for neigh_node in candidate_node.get_neighbours():
            if neigh_node in visited:
                continue

            visited.add(neigh_node)
            neigh_sim = cosine_similarity(search_vector, neigh_node.vector)
            if neigh_sim > out[-1][1] or len(out) < ef:
                candidates.append((neigh_node, neigh_sim))
                candidates.sort(key=lambda x: x[1])

                out.append((neigh_node, neigh_sim))
                out.sort(key=lambda x: x[1])

                if len(out) > ef:
                    out.pop(0)

    return out


def main():
    print("Starting")
    vectors = [random_vector(DIM) for _ in range(2000)]

    # Index
    print("Indexing")
    index = NSW(10, 32)

    t0 = time.time()
    for v in vectors:
        index.add_node(v)
    index_time = time.time() - t0

    print(f"Indexing took: {index_time:.5f} seconds")

    # Search
    print("Searching")

    search_vector = random_vector(DIM)

    # NSW
    t0 = time.time()
    nsw_sims = index.search(search_vector, ef=16)
    nsw_time = time.time() - t0

    best_node_sim = max(nsw_sims, key=lambda x: x[1])[1]
    print("NSW", best_node_sim, nsw_time)

    # Naive
    t0 = time.time()
    naive_best = max(cosine_similarity(search_vector, v) for v in vectors)
    naive_time = time.time() - t0

    print("Naive", naive_best, naive_time)


if __name__ == "__main__":
    main()
