# SPDX-License-Identifier: MPL-2.0
"""Clustering: exact, deterministic k-means (requires :doc:`numba <numba:index>`).

Labels equal those of plain float64 Lloyd from the same seeds, and results don't depend on the
thread count, threading layer or CPU.
"""

from __future__ import annotations

from ._kmeans import KMeansResult, kmeans, kmeans_plusplus


__all__ = ["KMeansResult", "kmeans", "kmeans_plusplus"]
