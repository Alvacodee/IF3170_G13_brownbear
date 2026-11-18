"""Log-loss contour and the (θ0, θ1) path of logistic regression training."""

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..models.logistic_regression import LogisticRegression
from ._style import CLASS_COLOR, FONT_SIZE, NODE_EDGE_COLOR, TREE_EDGE_COLOR

GRID_SIZE = 120
HOLD_SECONDS = 1.5
PATH_COLORS = (CLASS_COLOR, "#8fd18b", "#f2e394")


def loss_surface(
    model: LogisticRegression, x: np.ndarray, y: np.ndarray, theta0: np.ndarray, theta1: np.ndarray
) -> np.ndarray:
    """Objective of ``model`` for every (θ0, θ1) pair, indexed as [θ1, θ0].

    θ0 is the intercept and θ1 the weight of the single feature ``x``.
    """
    X = np.asarray(x, dtype=float).reshape(-1, 1)
    return np.array([[model.objective(X, y, coef=[slope], intercept=intercept) for intercept in theta0]
                     for slope in theta1])


def save_loss_path_video(
    models: dict[str, LogisticRegression],
    x: np.ndarray,
    y: np.ndarray,
    path: str | Path,
    fps: int = 6,
    max_frames: int = 150,
) -> Figure:
    """Write a video of training paths over the log-loss contour and return its last frame.

    Every model must share the objective, be fitted on the single feature ``x``, and use
    ``record_path=True``. Each frame advances all paths by one iteration, so paths
    that converge in fewer iterations stop earlier. Frames are taken on a logarithmic
    scale of the iteration number when the longest path exceeds ``max_frames``.
    """
    trajectories = {label: getattr(model, "path_", None) for label, model in models.items()}
    if any(t is None or t.shape[1] != 2 for t in trajectories.values()):
        raise ValueError("every model must be fitted on one feature with record_path=True")

    stacked = np.vstack(list(trajectories.values()))
    theta0, theta1 = _axis_range(stacked[:, 0]), _axis_range(stacked[:, 1])
    surface = loss_surface(next(iter(models.values())), x, y, theta0, theta1)

    figure, ax = plt.subplots(figsize=(8, 6), dpi=100)
    filled = ax.contourf(theta0, theta1, surface, levels=30, cmap="Blues_r", alpha=0.85)
    ax.contour(theta0, theta1, surface, levels=15, colors="white", linewidths=0.5, alpha=0.6)
    figure.colorbar(filled, ax=ax, label="log-loss")
    ax.plot(*stacked[0], marker="s", markersize=7, color=TREE_EDGE_COLOR, linestyle="", label="awal (0, 0)")
    artists = {}
    for (label, trajectory), color in zip(trajectories.items(), PATH_COLORS):
        line, = ax.plot([], [], color=color, linewidth=1.8, marker="o", markersize=3, label=label)
        point, = ax.plot([], [], marker="o", markersize=9, color=color, markeredgecolor=NODE_EDGE_COLOR)
        artists[label] = (line, point)
    ax.set_xlabel("θ0 (intercept)")
    ax.set_ylabel("θ1 (bobot fitur)")
    ax.legend(loc="upper right", fontsize=FONT_SIZE + 1)
    title = ax.set_title("")

    longest = max(len(t) for t in trajectories.values())
    frames = _frame_steps(longest, max_frames)
    with imageio.get_writer(path, fps=fps, macro_block_size=8) as writer:
        for step in list(frames) + [frames[-1]] * int(fps * HOLD_SECONDS):
            for label, trajectory in trajectories.items():
                last = min(step, len(trajectory) - 1)
                line, point = artists[label]
                line.set_data(trajectory[: last + 1, 0], trajectory[: last + 1, 1])
                point.set_data([trajectory[last, 0]], [trajectory[last, 1]])
            losses = ", ".join(f"{label} {models[label].loss_history_[min(step, len(t) - 1)]:.4f}"
                               for label, t in trajectories.items())
            title.set_text(f"Iterasi {step}  |  log-loss: {losses}")
            figure.canvas.draw()
            writer.append_data(np.asarray(figure.canvas.buffer_rgba())[..., :3])
    return figure


def _axis_range(values: np.ndarray, padding: float = 0.35, size: int = GRID_SIZE) -> np.ndarray:
    low, high = values.min(), values.max()
    margin = max(padding * (high - low), 0.5)
    return np.linspace(low - margin, high + margin, size)


def _frame_steps(n_steps: int, max_frames: int) -> np.ndarray:
    if n_steps <= max_frames:
        return np.arange(n_steps)
    return np.unique(np.round(np.geomspace(1, n_steps, max_frames)).astype(int) - 1)
