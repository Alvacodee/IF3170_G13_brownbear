"""Decision boundary, margins, and support vectors of SVM kernels in two dimensions."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from ..models.svm import SVM
from ._style import (
    FONT_SIZE,
    NEGATIVE_COLOR,
    NEGATIVE_POINT_COLOR,
    NODE_EDGE_COLOR,
    POSITIVE_COLOR,
    POSITIVE_POINT_COLOR,
)


def principal_components(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Project ``X`` onto its first principal components (SVD of the centered matrix)."""
    centered = np.asarray(X, dtype=float) - np.mean(X, axis=0)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    return centered @ components[:n_components].T


def balanced_sample(y: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """Row indices with ``per_class`` rows drawn from every class."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(np.flatnonzero(y == label), size=per_class, replace=False)
                           for label in np.unique(y)])


def plot_svm_kernels(
    X: np.ndarray,
    y: np.ndarray,
    kernels: dict[str, dict],
    per_class: int = 400,
    seed: int = 42,
    grid_size: int = 300,
) -> Figure:
    """Fit one SVM per kernel on two principal components and draw it.

    ``X`` is projected onto its first two principal components, then a class-balanced
    sample keeps the boundary from collapsing onto the majority class. In each panel
    the solid line is the decision boundary f(x) = 0, the dashed lines are the margins
    f(x) = -1 and f(x) = +1, and circled points are the support vectors.

    Parameters
    ----------
    kernels : dict
        Kernel name mapped to the other :class:`SVM` parameters, e.g.
        ``{"linear": {"C": 1}, "gaussian": {"C": 3, "gamma": 0.5}}``.
    """
    y = np.asarray(y)
    rows = balanced_sample(y, per_class, seed)
    points, labels = principal_components(X)[rows], y[rows]

    low, high = points.min(axis=0) - 0.5, points.max(axis=0) + 0.5
    xx, yy = np.meshgrid(np.linspace(low[0], high[0], grid_size), np.linspace(low[1], high[1], grid_size))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    regions = ListedColormap([NEGATIVE_COLOR, POSITIVE_COLOR])
    point_colors = np.where(labels == labels.max(), POSITIVE_POINT_COLOR, NEGATIVE_POINT_COLOR)

    figure, axes = plt.subplots(1, len(kernels), figsize=(5.6 * len(kernels), 5.2), sharey=True)
    for ax, (kernel, params) in zip(np.atleast_1d(axes), kernels.items()):
        model = SVM(kernel=kernel, **params).fit(points, labels)
        scores = model.decision_function(grid).reshape(xx.shape)
        ax.contourf(xx, yy, scores > 0, levels=[-0.5, 0.5, 1.5], cmap=regions, alpha=0.55)
        ax.contour(xx, yy, scores, levels=[-1, 0, 1], colors=NODE_EDGE_COLOR,
                   linestyles=["--", "-", "--"], linewidths=[1, 1.6, 1])
        ax.scatter(*points.T, c=point_colors, s=9, linewidths=0, alpha=0.8)
        support = model.support_vectors_
        ax.scatter(*support.T, s=34, facecolors="none", edgecolors=NODE_EDGE_COLOR, linewidths=0.6)
        ax.set_title(f"Kernel {kernel}: {len(support)} support vector", fontsize=FONT_SIZE + 2)
        ax.set_xlabel("PC1")
    np.atleast_1d(axes)[0].set_ylabel("PC2")

    figure.legend(handles=[
        Line2D([], [], color=NODE_EDGE_COLOR, linewidth=1.6, label="bidang pemisah f(x) = 0"),
        Line2D([], [], color=NODE_EDGE_COLOR, linestyle="--", label="margin f(x) = ±1"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=NODE_EDGE_COLOR,
               label="support vector"),
        Line2D([], [], marker="o", linestyle="", color=NEGATIVE_POINT_COLOR, label="tidak batal"),
        Line2D([], [], marker="o", linestyle="", color=POSITIVE_POINT_COLOR, label="batal"),
    ], loc="lower center", ncol=5, frameon=False, fontsize=FONT_SIZE + 1)
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    return figure
