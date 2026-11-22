"""Colors, sizes, and drawing helpers shared by every figure."""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch

CLASS_COLOR = "#f4a261"
FEATURE_COLOR = "#dbe9f6"
NODE_EDGE_COLOR = "#4a4a4a"
TREE_EDGE_COLOR = "#1d6fa5"
CLASS_EDGE_COLOR = "#9a9a9a"
MUTED_TEXT_COLOR = "#6a6a6a"
DASHED = (0, (4, 3))
FONT_SIZE = 8.5

# Kelas 0 (tidak batal) biru, kelas 1 (batal) oranye.
NEGATIVE_COLOR = np.array([0.86, 0.91, 0.96])
POSITIVE_COLOR = np.array([0.96, 0.64, 0.38])


def draw_arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    linewidth: float,
    linestyle: str | tuple = "-",
) -> None:
    """Draw an arrow whose ends touch ``start`` and ``end`` exactly."""
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=8 + 2 * linewidth, shrinkA=0, shrinkB=0,
        color=color, linewidth=linewidth, linestyle=linestyle, zorder=2,
    ))
