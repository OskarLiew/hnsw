import random
import time

from common import random_vector, cosine_similarity

K = 10
DIM = 8


class Node:
    def __init__(
        self, vector: list[float], neighbours: list[tuple["Node", float]] | None = None
    ) -> None:
        self.vector = vector
        self.neighbours = neighbours or []


def main():
    print("Starting")
    vectors = [random_vector(DIM) for _ in range(1000)]

    print("Indexing")
    t0 = time.time()
    # Index
    nodes: list[Node] = []
    for v in vectors:
        node_sims = sorted(
            nsw_search(nodes, v, iters=10).items(),
            key=lambda node_sim: -node_sim[1],
        )

        new_node = Node(v, neighbours=node_sims[:K])
        for n, sim in node_sims[:K]:
            n.neighbours = sorted(
                n.neighbours + [(new_node, sim)], key=lambda node_sim: -node_sim[1]
            )[:K]

        nodes.append(new_node)

    index_time = time.time() - t0
    print(f"Indexing took: {index_time:.5f} seconds")

    # Search
    search_vector = random_vector(DIM)

    print("Searching")

    # NSW
    t0 = time.time()
    nsw_sims = nsw_search(nodes, search_vector, iters=10)
    nsw_time = time.time() - t0

    best_node_sim = max(nsw_sims.items(), key=lambda x: x[1])[1]
    print("NSW", best_node_sim, nsw_time)

    # Naive
    t0 = time.time()
    naive_best = max(cosine_similarity(search_vector, v) for v in vectors)
    naive_time = time.time() - t0

    print("Naive", naive_best, naive_time)


def nsw_search(
    nodes: list[Node], search_vector: list[float], iters: int = 10
) -> dict[Node, float]:
    if not nodes:
        return {}

    def search():
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

    sims = {}
    for _ in range(iters):
        sims.update(search())

    return sims


if __name__ == "__main__":
    main()
