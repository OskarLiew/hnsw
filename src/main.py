import math
import random
import time

K = 10
DIM = 32


class Node:
    def __init__(
        self, vector: list[float], neighbours: list[tuple["Node", float]] | None = None
    ) -> None:
        self.vector = vector
        self.neighbours = neighbours or []


def main():
    def random_vector(size: int) -> list[float]:
        return [random.random() for _ in range(size)]

    vectors = [random_vector(DIM) for _ in range(1000)]

    nodes: list[Node] = []
    for v in vectors:
        node_sims = sorted(
            [(n, cosine_similarity(v, n.vector)) for n in nodes],
            key=lambda node_sim: -node_sim[1],
        )

        new_node = Node(v, neighbours=node_sims[:K])
        for n, sim in node_sims[:K]:
            n.neighbours = sorted(
                n.neighbours + [(new_node, sim)], key=lambda node_sim: -node_sim[1]
            )[:K]

        nodes.append(new_node)

    search_vector = random_vector(DIM)

    t0 = time.time()
    best_node_sim = 0
    for _ in range(10):
        node = random.choice(nodes)
        while True:
            sim = cosine_similarity(search_vector, node.vector)
            max_neigbour, max_neigbour_sim = max(
                [
                    (n, cosine_similarity(search_vector, n.vector))
                    for n, _ in node.neighbours
                ],
                key=lambda node_sim: -node_sim[1],
            )
            if max_neigbour_sim > sim:
                node = max_neigbour
            else:
                break

        if sim > best_node_sim:
            best_node_sim = sim
    nsw_time = time.time() - t0

    print("NSW", best_node_sim, nsw_time)

    t0 = time.time()
    naive_best = max(cosine_similarity(search_vector, v) for v in vectors)
    naive_time = time.time() - t0

    print("Naive", naive_best, naive_time)


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    assert len(v1) == len(v2)

    m1 = vector_norm(v1)
    m2 = vector_norm(v2)
    return sum(e1 * e2 / (m1 * m2) for e1, e2 in zip(v1, v2))


def vector_norm(v: list[float]) -> float:
    return math.sqrt(sum(e**2 for e in v))


if __name__ == "__main__":
    main()
