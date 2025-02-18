import time

import tqdm
from hnsw_ef import HNSWEF
from nsw_ef import NSWEF
import numpy as np
import scipy.stats


from nsw import NSW
from hnsw import HNSW
from common import random_vector, cosine_similarity

DIM = 32

N_SAMPLES = 100


def main():
    vectors = [random_vector(DIM) for _ in range(10000)]
    search_vector = random_vector(DIM)

    # Naive
    naive_results = [
        naive(search_vector, vectors)
        for _ in tqdm.trange(N_SAMPLES, desc="Naive search")
    ]
    naive_times = np.array([x[1] for x in naive_results])
    mean, diff = mean_and_interval(naive_times, confidence=0.95)
    print(
        f"Similarity: {naive_results[0][0]:.4f}\n"
        f"Runtime: {mean:.4f} +/- {diff:.3e} seconds\n"
    )

    # NSW
    n_edges = 16
    nsw_index = NSW(n_edges=n_edges)
    for vector in tqdm.tqdm(vectors, desc="Indexing NSW"):
        nsw_index.add_node(vector)
    nsw_results = [
        index_search(search_vector, nsw_index)
        for _ in tqdm.trange(N_SAMPLES, desc="NSW search")
    ]
    nsw_sims = np.array([x[0] for x in nsw_results])
    nsw_times = np.array([x[1] for x in nsw_results])
    mean_sims, diff_sims = mean_and_interval(nsw_sims, confidence=0.95)
    mean_time, diff_time = mean_and_interval(nsw_times, confidence=0.95)
    print(
        f"Similarity: {mean_sims:.4f} +/- {diff_sims:.3e}, median {np.median(nsw_sims):.4f}\n"
        f"Runtime: {mean_time:.4f} +/- {diff_time:.3e} seconds\n"
    )

    # HNSW
    hnsw_index = HNSW(
        n_layers=4,
        n_edges=8,
    )
    for vector in tqdm.tqdm(vectors, desc="Indexing HNSW"):
        hnsw_index.add_node(vector)

    hnsw_results = [
        index_search(search_vector, hnsw_index)
        for _ in tqdm.trange(N_SAMPLES, desc="HNSW search")
    ]
    hnsw_sims = np.array([x[0] for x in hnsw_results])
    hnsw_times = np.array([x[1] for x in hnsw_results])
    mean_sims, diff_sims = mean_and_interval(hnsw_sims, confidence=0.95)
    mean_time, diff_time = mean_and_interval(hnsw_times, confidence=0.95)
    print(
        f"Similarity: {mean_sims:.4f} +/- {diff_sims:.3e}, median {np.median(nsw_sims):.4f}\n"
        f"Runtime: {mean_time:.4f} +/- {diff_time:.3e} seconds\n"
    )

    # NSWEF
    n_edges = 16
    nsw_index = NSWEF(n_edges=n_edges, ef_construct=32)
    for vector in tqdm.tqdm(vectors, desc="Indexing NSW w/ priority queue"):
        nsw_index.add_node(vector)
    nsw_results = [
        index_search(search_vector, nsw_index)
        for _ in tqdm.trange(N_SAMPLES, desc="NSW w/ priority queue search")
    ]
    nsw_sims = np.array([x[0] for x in nsw_results])
    nsw_times = np.array([x[1] for x in nsw_results])
    mean_sims, diff_sims = mean_and_interval(nsw_sims, confidence=0.95)
    mean_time, diff_time = mean_and_interval(nsw_times, confidence=0.95)
    print(
        f"Similarity: {mean_sims:.4f} +/- {diff_sims:.3e}, median {np.median(nsw_sims):.4f}\n"
        f"Runtime: {mean_time:.4f} +/- {diff_time:.3e} seconds\n"
    )

    # HNSWEF
    hnsw_index = HNSWEF(
        n_layers=4,
        n_edges=8,
        ef_construct=32,
    )
    for vector in tqdm.tqdm(vectors, desc="Indexing HNSW w/ priority queue"):
        hnsw_index.add_node(vector)

    hnsw_results = [
        index_search(search_vector, hnsw_index)
        for _ in tqdm.trange(N_SAMPLES, desc="HNSW search w/ priority queue")
    ]
    hnsw_sims = np.array([x[0] for x in hnsw_results])
    hnsw_times = np.array([x[1] for x in hnsw_results])
    mean_sims, diff_sims = mean_and_interval(hnsw_sims, confidence=0.95)
    mean_time, diff_time = mean_and_interval(hnsw_times, confidence=0.95)
    print(
        f"Similarity: {mean_sims:.4f} +/- {diff_sims:.3e}, median {np.median(nsw_sims):.4f}\n"
        f"Runtime: {mean_time:.4f} +/- {diff_time:.3e} seconds\n"
    )


def naive(
    search_vector: list[float], vectors: list[list[float]]
) -> tuple[float, float]:
    t0 = time.time()
    best_sim = max(cosine_similarity(search_vector, v) for v in vectors)
    runtime = time.time() - t0
    return best_sim, runtime


def index_search(search_vector: list[float], index) -> tuple[float, float]:
    t0 = time.time()
    node_sims = index.search(search_vector)
    runtime = time.time() - t0
    return node_sims[0][1], runtime


def mean_and_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, h


if __name__ == "__main__":
    main()
