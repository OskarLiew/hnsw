import random
import time

from common import random_vector, cosine_similarity

K = 5
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


class HNSW:
    def __init__(self, m_l: int, p_l: float = 0.2) -> None:
        self.layers: list[list[Node]] = [[] for _ in range(m_l)]
        self.p_l = p_l

    def add_node(self, vector: list[float]) -> None:
        parent = None
        for i in range(len(self.layers)):
            parent = self._add_to_layer(vector, i, parent)
            if (  # Add to next layer if it's empty, otherwise with probability p_l
                i + 1 < len(self.layers) and self.layers[i + 1]
            ) and random.random() > self.p_l:
                break

    def _add_to_layer(
        self, vector: list[float], layer_idx: int, parent: Node | None
    ) -> Node:
        layer = self.layers[layer_idx]
        node_sims = self._nsw_search(vector, layer)

        new_node = Node(vector, neighbours=node_sims[:K], parent=parent)
        for n, sim in node_sims[:K]:
            n.neighbours = sorted(
                n.neighbours + [(new_node, sim)], key=lambda node_sim: -node_sim[1]
            )[:K]

        self.layers[layer_idx].append(new_node)
        return new_node

    def _nsw_search(
        self,
        search_vector: list[float],
        nodes: list[Node],
        start_node: Node | None = None,
        iters: int = 10,
    ) -> list[tuple[Node, float]]:
        if not nodes:
            return []

        def search():
            sims = {}
            node = start_node or random.choice(nodes)
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

        sims: dict[Node, float] = {}
        for _ in range(iters):
            sims.update(search())

        return sorted(
            sims.items(),
            key=lambda node_sim: -node_sim[1],
        )

    def search(
        self, search_vector: list[float], iters: int = 10
    ) -> list[tuple[Node, float]]:
        def search():
            sims = {}
            start_node = None
            for layer in reversed(self.layers):
                node_sims = self._nsw_search(search_vector, layer, start_node, iters=1)

                best_node = node_sims[0][0]
                if not best_node.parent:
                    break

                start_node = best_node.parent
                sims.update(node_sims)

            return sims

        sims: dict[Node, float] = {}
        for _ in range(iters):
            sims.update(search())

        return sorted(
            sims.items(),
            key=lambda node_sim: -node_sim[1],
        )


def main():
    print("Starting")
    vectors = [random_vector(DIM) for _ in range(2000)]

    print("Indexing")
    t0 = time.time()
    # Index
    index = HNSW(m_l=3, p_l=0.3)
    for v in vectors:
        index.add_node(v)

    index_time = time.time() - t0
    print(f"Indexing took: {index_time:.5f} seconds")

    # Search
    search_vector = random_vector(DIM)

    print("Searching")

    # NSW
    t0 = time.time()
    hnsw_sims = index.search(search_vector)
    hnsw_time = time.time() - t0

    best_node_sim = max(hnsw_sims, key=lambda x: x[1])[1]
    print("HNSW", best_node_sim, hnsw_time)

    # Naive
    t0 = time.time()
    naive_best = max(cosine_similarity(search_vector, v) for v in vectors)
    naive_time = time.time() - t0

    print("Naive", naive_best, naive_time)


if __name__ == "__main__":
    main()
