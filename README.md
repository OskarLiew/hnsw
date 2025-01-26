# Hierarcical Navigable Small World

This reposotiry imlements the HNSW algorithm in pure python as a learning
exercise in how vector databases perform indexing for faster nearest neighbour
search.

## Run the examples

The file `src/main.py` contains the evaluation script that compares naive
search, NSW and HNSW. It can be run after installing the dependencies from
`requirements.txt`.

## Explanation

The naive approach of comparing a search vector to all other vectors grows
linearly with the number of vectors `O(N)` in the search set. When `N` grows
large the calculation may become infeasibly slow for many applications.

To combat this, a common solution is to use an approximate nearest neighbour
algorithm. A common choice that is used by most mainstream vector databases is
Hierarcical Navigable Small World (HNSW).

It constructs several graphs of nearest neighbours with an increasing number of
nodes in each graph. Search starts in the top layer, which contains the fewest
nodes, and the graph is traversed until a local minimum is found. At that point
the algorithm moves on to the next layer, which contains a larger number of
nodes. This goes on until the minimum at the bottom layer is found, which is the
approximate nearest neighbour. Unlike the naive approach, the time complexity of
HNSW is `O(log N)`. The main drawback is that the index must be built before
searching.
