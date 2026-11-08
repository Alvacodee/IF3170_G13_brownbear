from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from .models import SVM

KERNEL_PARAMS = {
    "linear": {},
    "polynomial": {"degree": 3, "coef0": 1.0},
    "gaussian": {},
}


def plot_kernels(X: np.ndarray, y: np.ndarray, path: Path, per_class: int = 500, seed: int = 42) -> None:
    # Sampel seimbang supaya di 2D tidak semua titik diprediksi sebagai kelas mayoritas.
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    rows = np.concatenate([rng.choice(np.flatnonzero(y == label), size=per_class, replace=False) for label in (0, 1)])
    points = PCA(n_components=2, random_state=seed).fit_transform(X)[rows]
    labels = y[rows]

    low, high = points.min(axis=0) - 0.5, points.max(axis=0) + 0.5
    xx, yy = np.meshgrid(np.linspace(low[0], high[0], 300), np.linspace(low[1], high[1], 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for ax, (kernel, params) in zip(axes, KERNEL_PARAMS.items()):
        model = SVM(kernel=kernel, **params).fit(points, labels)
        scores = model.decision_function(grid).reshape(xx.shape)

        ax.contourf(xx, yy, scores > 0, levels=[-0.5, 0.5, 1.5], cmap="coolwarm", alpha=0.2)
        ax.contour(xx, yy, scores, levels=[-1, 0, 1], colors="black", linestyles=["--", "-", "--"], linewidths=1)
        ax.scatter(points[:, 0], points[:, 1], c=labels, cmap="coolwarm", s=8, edgecolors="none")
        support = model.support_vectors_
        ax.scatter(support[:, 0], support[:, 1], s=36, facecolors="none", edgecolors="black", linewidths=0.5)
        ax.set_title(f"{kernel.capitalize()} kernel, {len(support)} support vectors")
        ax.set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    fig.suptitle("SVM boundary (solid) and margin (dashed) on 2D PCA, balanced sample, red = canceled")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
