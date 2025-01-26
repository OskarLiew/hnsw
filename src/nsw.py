import random
import time

from common import random_vector, cosine_similarity

K = 5
DIM = 32


class Node:
    def __init__(
        self, vector: list[float], neighbours: list[tuple["Node", float]] | None = None
    ) -> None:
        self.vector = vector
        self.neighbours = neighbours or []


class NSW:
    def __init__(self) -> None:
        self.nodes = []

    def add_node(self, vector: list[float]) -> None:
        node_sims = self.search(vector)

        new_node = Node(vector, neighbours=node_sims[:K])
        for n, sim in node_sims[:K]:
            n.neighbours = sorted(
                n.neighbours + [(new_node, sim)], key=lambda node_sim: -node_sim[1]
            )[:K]

        self.nodes.append(new_node)

    def search(
        self, search_vector: list[float], iters: int = 10
    ) -> list[tuple[Node, float]]:
        if not self.nodes:
            return []

        sims = {}
        for _ in range(iters):
            sims.update(nsw_search(search_vector, self.nodes))

        return sorted(
            sims.items(),
            key=lambda node_sim: -node_sim[1],
        )


def nsw_search(search_vector: list[float], nodes: list[Node]) -> dict[Node, float]:
    sims = {}
    node = random.choice(nodes)
    while True:
        if node not in sims:
            sims[node] = cosine_similarity(search_vector, node.vector)

        for n, _ in node.neighbours:
            if n in sims:
                continue
            sims[n] = cosine_similarity(search_vector, n.vector)

        max_node, _ = max(sims.items(), key=lambda x: x[1])
        if max_node == node:
            return sims
        node = max_node


def main():
    print("Starting")
    vectors = [random_vector(DIM) for _ in range(2000)]

    # Index
    print("Indexing")
    index = NSW()

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
    nsw_sims = index.search(search_vector)
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
