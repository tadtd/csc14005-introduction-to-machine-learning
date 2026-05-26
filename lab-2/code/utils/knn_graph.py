"""kNN graph for neighbor embedding (used when passing ``graph=`` to CNE)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse import lil_matrix
from sklearn.neighbors import NearestNeighbors


def symmetrized_knn_graph(X: np.ndarray, k: int = 15, n_jobs: int = -1) -> sp.csr_matrix:
    """
    Undirected kNN adjacency (unweighted), symmetrized on the fly.

    Used by ``NegTSNE`` before calling ``cne.CNE``.
    """
    X = np.asarray(X, dtype=np.float32)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=n_jobs)
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    adj = lil_matrix((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        neighs = indices[i, 1:]
        adj[i, neighs] = 1
        adj[neighs, i] = 1
    return adj.tocsr()
