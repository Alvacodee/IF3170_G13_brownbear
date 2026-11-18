"""Top-N splits of a fitted decision tree."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch

from ..models.dtl import DecisionTree
from ._style import (
    DASHED,
    FONT_SIZE,
    NEGATIVE_COLOR,
    NODE_EDGE_COLOR,
    POSITIVE_COLOR,
    TREE_EDGE_COLOR,
    draw_arrow,
)

TREE_BOX_WIDTH = 2.1
TREE_BOX_HEIGHT = 1.12
TREE_LEAF_GAP = 0.25
TREE_LEVEL_HEIGHT = 1.55
MAX_CATEGORIES_SHOWN = 2


def select_top_splits(model: DecisionTree, top_n: int) -> list[int]:
    """Return the ``top_n`` most important splits that stay connected to the root.

    Splits are expanded best-first from the root by their weighted Gini decrease,
    so the figure always shows the splits that reduce impurity the most.
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    weights = model.value_.sum(axis=1)

    def decrease(node: int) -> float:
        left, right = model.children_left_[node], model.children_right_[node]
        return (weights[node] * model.impurity_[node] - weights[left] * model.impurity_[left]
                - weights[right] * model.impurity_[right])

    chosen, frontier = [], [0] if model.feature_[0] >= 0 else []
    while frontier and len(chosen) < top_n:
        node = max(frontier, key=decrease)
        frontier.remove(node)
        chosen.append(node)
        frontier.extend(child for child in (model.children_left_[node], model.children_right_[node])
                        if model.feature_[child] >= 0)
    return chosen


def plot_decision_tree(
    model: DecisionTree,
    feature_names: list[str] | None = None,
    top_n: int = 10,
    class_names: tuple[str, str] = ("tidak batal", "batal"),
    ax: Axes | None = None,
) -> Axes:
    """Draw the ``top_n`` most important splits of a fitted :class:`DecisionTree`.

    Each box shows the split condition, Gini (with class weights, if any), sample count, and the
    actual share of the second class in the training rows that reach the node. Box color moves
    from blue to orange as that share grows. Subtrees that are not
    among the top-N splits are collapsed into a dashed box. Missing values follow the
    branch marked ``NaN``.
    """
    chosen = set(select_top_splits(model, top_n))
    children = {node: [int(model.children_left_[node]), int(model.children_right_[node])] for node in chosen}

    positions: dict[int, tuple[float, float]] = {}
    n_leaves = 0

    def place(node: int, depth: int) -> None:
        nonlocal n_leaves
        if node in children:
            for child in children[node]:
                place(child, depth + 1)
            x = float(np.mean([positions[child][0] for child in children[node]]))
        else:
            x, n_leaves = n_leaves * (TREE_BOX_WIDTH + TREE_LEAF_GAP), n_leaves + 1
        positions[node] = (x, -depth * TREE_LEVEL_HEIGHT)

    place(0, 0)
    xs = [x for x, _ in positions.values()]
    ys = [y for _, y in positions.values()]
    x_limits = (min(xs) - TREE_BOX_WIDTH / 2 - 0.2, max(xs) + TREE_BOX_WIDTH / 2 + 0.2)
    y_limits = (min(ys) - TREE_BOX_HEIGHT / 2 - 0.2, TREE_BOX_HEIGHT / 2 + 0.6)
    if ax is None:
        figure = plt.figure(figsize=(max(6.0, 0.62 * np.ptp(x_limits)), max(3.0, 0.62 * np.ptp(y_limits))))
        ax = figure.add_axes((0, 0, 1, 1))

    names = feature_names if feature_names is not None else model.feature_names_in_
    total = model.value_[0].sum()
    for parent, (left, right) in children.items():
        (xp, yp) = positions[parent]
        nan_left = bool(model.missing_left_[parent])
        for child, label in ((left, "ya" + (" / NaN" if nan_left else "")),
                             (right, "tidak" + ("" if nan_left else " / NaN"))):
            xc, yc = positions[child]
            share = model.value_[child].sum() / total
            draw_arrow(ax, (xp, yp - TREE_BOX_HEIGHT / 2), (xc, yc + TREE_BOX_HEIGHT / 2),
                        color=TREE_EDGE_COLOR, linewidth=0.8 + 4 * share)
            ax.text((xp + xc) / 2, (yp + yc) / 2 + 0.05, label, fontsize=FONT_SIZE - 1, ha="center",
                    va="center", color=TREE_EDGE_COLOR, zorder=4,
                    bbox={"boxstyle": "square,pad=0.1", "facecolor": "white", "edgecolor": "none"})

    for node, (x, y) in positions.items():
        counts = model.node_class_counts_[node]
        positive = counts[-1] / counts.sum()
        color = NEGATIVE_COLOR + (POSITIVE_COLOR - NEGATIVE_COLOR) * positive
        stats = (f"gini={model.impurity_[node]:.3f}  n={model.n_node_samples_[node]:,}\n"
                 f"{class_names[-1]}={positive:.0%}")
        if node in children:
            label = _short_split(model, node, names) + "\n" + stats
            style = "-"
        elif model.feature_[node] >= 0:
            label = f"subpohon lanjutan\n{stats}"
            style = DASHED
        else:
            predicted = class_names[int(np.argmax(model.value_[node]))]
            label = f"daun: {predicted}\n{stats}"
            style = "-"
        ax.add_patch(FancyBboxPatch(
            (x - TREE_BOX_WIDTH / 2, y - TREE_BOX_HEIGHT / 2), TREE_BOX_WIDTH, TREE_BOX_HEIGHT,
            boxstyle="round,pad=0,rounding_size=0.08", facecolor=color, edgecolor=NODE_EDGE_COLOR,
            linewidth=1.3 if node in children else 0.9, linestyle=style, zorder=3,
        ))
        ax.text(x, y, label, ha="center", va="center", fontsize=FONT_SIZE - 1, linespacing=1.15, zorder=4)

    ax.text(np.mean(x_limits), y_limits[1] - 0.25,
            f"{len(children)} split teratas dari {int(np.sum(model.feature_ >= 0))} split; "
            f"warna = proporsi {class_names[-1]}", ha="center", fontsize=FONT_SIZE, color=NODE_EDGE_COLOR)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def _short_split(model: DecisionTree, node: int, names) -> str:
    feature = int(model.feature_[node])
    name = str(names[feature]) if names is not None else f"x[{feature}]"
    if model.is_categorical_[feature]:
        chosen = [str(c) for c in model.categories_[feature][model.left_categories_[node]]]
        extra = len(chosen) - MAX_CATEGORIES_SHOWN
        shown = chosen[:MAX_CATEGORIES_SHOWN]
        if extra > 0:
            shown[-1] += f", +{extra} lainnya"
        return f"{name} in\n{{" + ",\n".join(shown) + "}"
    return f"{name}\n<= {model.threshold_[node]:.4g}"
