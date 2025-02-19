import time

import tqdm
import numpy as np
import scipy.stats


from hnsw_paper import HNSW
from common import random_vector, cosine_distance

DIM = 32

N_SAMPLES = 100


def main():
    vectors = [random_vector(DIM) for _ in range(1000)]
    search_vector = random_vector(DIM)

    # Naive
    naive_results = [
        naive(search_vector, vectors)
        for _ in tqdm.trange(N_SAMPLES, desc="Naive search")
    ]
    naive_times = np.array([x[1] for x in naive_results])
    mean, diff = mean_and_interval(naive_times, confidence=0.95)
    print(
        f"Distance: {naive_results[0][0]:.4f}\n"
        f"Runtime: {mean:.4f} +/- {diff:.3e} seconds\n"
    )

    ### HNSW
    print("Indexing")
    t0 = time.time()
    index = HNSW(
        n_layers=4,
        m_max=12,
        ef_construction=64,
        m_l=None,
        prune_connections=False,
        extend_candidates=False,
    )
    for v in tqdm.tqdm(vectors, desc="HNSW indexing"):
        index.insert(v)

    index_time = time.time() - t0
    print(f"Indexing took: {index_time:.5f} seconds")

    print("Graph layers")
    for l, d in index.neighbours.items():
        print(l, len(d))

    # Search
    hnsw_results = [
        index_search(search_vector, index, ef=16)
        for _ in tqdm.trange(N_SAMPLES, desc="HNSW search")
    ]

    hnsw_dists = np.array([x[0] for x in hnsw_results])
    hnsw_times = np.array([x[1] for x in hnsw_results])
    mean_dists, diff_dists = mean_and_interval(hnsw_dists, confidence=0.95)
    mean_time, diff_time = mean_and_interval(hnsw_times, confidence=0.95)
    print(
        f"Distance: {mean_dists:.4f} +/- {diff_dists:.3e}, median {np.median(hnsw_dists):.4f}\n"
        f"Runtime: {mean_time:.4f} +/- {diff_time:.3e} seconds\n"
    )

    # NSW
    index.n_layers = 1
    # Search
    nsw_results = [
        index_search(search_vector, index, ef=16)
        for _ in tqdm.trange(N_SAMPLES, desc="NSW search")
    ]

    nsw_dists = np.array([x[0] for x in nsw_results])
    nsw_times = np.array([x[1] for x in nsw_results])
    mean_dists, diff_dists = mean_and_interval(nsw_dists, confidence=0.95)
    mean_time, diff_time = mean_and_interval(nsw_times, confidence=0.95)
    print(
        f"Distance: {mean_dists:.4f} +/- {diff_dists:.3e}, median {np.median(nsw_dists):.4f}\n"
        f"Runtime: {mean_time:.4f} +/- {diff_time:.3e} seconds\n"
    )


def naive(
    search_vector: list[float], vectors: list[list[float]]
) -> tuple[float, float]:
    t0 = time.time()
    best_dists = min(cosine_distance(search_vector, v) for v in vectors)
    runtime = time.time() - t0
    return best_dists, runtime


def index_search(
    search_vector: list[float], index: HNSW, **kwargs
) -> tuple[float, float]:
    t0 = time.time()
    node_dists = index.search(search_vector, **kwargs)
    runtime = time.time() - t0
    return node_dists[0][0], runtime


def mean_and_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, h


if __name__ == "__main__":
    main()
