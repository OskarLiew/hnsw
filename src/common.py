import math
import random


def random_vector(size: int) -> list[float]:
    return [random.random() for _ in range(size)]


def cosine_distance(v1: list[float], v2: list[float]) -> float:
    assert len(v1) == len(v2)

    m1 = vector_norm(v1)
    m2 = vector_norm(v2)
    return -sum(e1 * e2 / (m1 * m2) for e1, e2 in zip(v1, v2))


def vector_norm(v: list[float]) -> float:
    return math.sqrt(sum(e**2 for e in v))
