"""MP4 of one-feature Logistic Regression training on a log-loss contour."""

from pathlib import Path

import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch


def save_logistic_regression_animation(
    x: np.ndarray,
    y: np.ndarray,
    parameter_history: list[tuple[float, float]],
    loss_history: list[float],
    output_path: Path,
) -> None:
    """Export an unweighted, unregularized training trace using its training data."""
    path = np.asarray(parameter_history)
    bias, weight = path.T
    padding = 0.35 * np.ptp(path, axis=0) + 0.08
    bias_grid = np.linspace(bias.min() - padding[0], bias.max() + padding[0], 101)
    weight_grid = np.linspace(weight.min() - padding[1], weight.max() + padding[1], 101)
    loss_grid = np.empty((weight_grid.size, bias_grid.size))
    for row, theta_1 in enumerate(weight_grid):
        z = bias_grid[:, None] + theta_1 * x
        loss_grid[row] = (np.logaddexp(0, z) - y * z).mean(axis=1)

    background = "#0b1425"
    foreground = "#f4f7fc"
    muted = "#b2c4dd"
    accent = "#ffb565"
    colors = LinearSegmentedColormap.from_list(
        "log_loss", ["#def9f4", "#80d4d5", "#367eaa", "#292861"],
    )
    levels = np.linspace(loss_grid.min(), loss_grid.max(), 25)
    fig = plt.figure(figsize=(12.8, 7.2), facecolor=background)
    try:
        ax = fig.add_axes((0.09, 0.17, 0.53, 0.62))
        color_ax = fig.add_axes((0.64, 0.17, 0.016, 0.62))
        filled = ax.contourf(bias_grid, weight_grid, loss_grid, levels=levels, cmap=colors)
        ax.contour(bias_grid, weight_grid, loss_grid, levels=levels[2::3],
                   colors="#172552", linewidths=0.65, alpha=0.41)
        colorbar = fig.colorbar(filled, cax=color_ax, ticks=levels[::6], format="%.3f")
        colorbar.set_label("Log-loss", color=foreground, labelpad=10)
        colorbar.ax.tick_params(colors=muted, labelsize=10)
        colorbar.outline.set_edgecolor(muted)
        ax.set_xlabel(r"Bias $\theta_0$", color=foreground, fontsize=13, labelpad=9)
        ax.set_ylabel(r"Bobot fitur $\theta_1$", color=foreground, fontsize=13, labelpad=9)
        ax.tick_params(colors=muted, labelsize=11)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_color(muted)

        fig.text(0.09, 0.902, "Lintasan optimasi Logistic Regression",
                 color=foreground, fontsize=28, fontweight="bold")
        fig.text(0.09, 0.852, "days_before_arrival terstandardisasi  /  FISTA tanpa regularisasi",
                 color=muted, fontsize=13)
        fig.text(0.09, 0.047, "Setiap garis kontur menghubungkan parameter dengan log-loss yang sama.",
                 color=muted, fontsize=11)

        panel = fig.add_axes((0.76, 0.17, 0.185, 0.62))
        panel.set_axis_off()
        panel.add_patch(FancyBboxPatch(
            (0, 0), 1, 1, boxstyle="round,pad=0.008,rounding_size=0.035",
            transform=panel.transAxes, facecolor="#15233a", edgecolor="#334965", clip_on=False,
        ))
        panel.text(0.10, 0.87, "Langkah diterima", color=muted, fontsize=13)
        step_text = panel.text(0.10, 0.75, "", color=accent, fontsize=26, fontfamily="monospace")
        panel.text(0.10, 0.61, "Log-loss", color=muted, fontsize=13)
        loss_text = panel.text(0.10, 0.49, "", color=foreground, fontsize=23, fontfamily="monospace")
        panel.plot([0.10, 0.90], [0.41, 0.41], color="#40556e", transform=panel.transAxes)
        panel.text(0.10, 0.30, r"Bias $\theta_0$", color=muted, fontsize=12)
        panel.text(0.10, 0.19, r"Bobot $\theta_1$", color=muted, fontsize=12)
        bias_text = panel.text(0.90, 0.30, "", color=foreground, fontsize=12, ha="right",
                               fontfamily="monospace")
        weight_text = panel.text(0.90, 0.19, "", color=foreground, fontsize=12, ha="right",
                                 fontfamily="monospace")
        panel.set_xlim(0, 1)
        panel.set_ylim(0, 1)

        line, = ax.plot([], [], color=accent, linewidth=2.6, zorder=4, path_effects=[
            patheffects.withStroke(linewidth=4.6, foreground=background),
        ])
        ax.plot([bias[0]], [weight[0]], marker="D", markersize=7, markerfacecolor=foreground,
                markeredgecolor=background, zorder=5)
        ax.annotate("Awal", (bias[0], weight[0]), xytext=(9, -16), textcoords="offset points",
                    color=background, fontsize=10, fontweight="bold", zorder=6,
                    path_effects=[patheffects.withStroke(linewidth=2.8, foreground=foreground)])
        point, = ax.plot([], [], marker="o", markersize=10, markerfacecolor=accent,
                         markeredgecolor=background, markeredgewidth=1.4, zorder=7)
        last_step = len(path) - 1

        def update(frame):
            line.set_data(bias[:frame + 1], weight[:frame + 1])
            point.set_data(bias[frame:frame + 1], weight[frame:frame + 1])
            step_text.set_text(f"{frame} / {last_step}")
            loss_text.set_text(f"{loss_history[frame]:.6f}")
            bias_text.set_text(f"{bias[frame]:+.4f}")
            weight_text.set_text(f"{weight[frame]:+.4f}")

        fps = 6
        frames = [0] * fps + list(range(1, last_step)) + [last_step] * (2 * fps)
        animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps)
        with plt.rc_context({"animation.ffmpeg_path": imageio_ffmpeg.get_ffmpeg_exe()}):
            writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=[
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            ])
            animation.save(output_path, writer=writer, dpi=100)
    finally:
        plt.close(fig)
