"""Figures for the trained models."""

from .bayesian_network import plot_bayesian_network
from .decision_tree import plot_decision_tree, select_top_splits
from .logistic_regression import loss_surface, save_loss_path_video
from .svm import plot_svm_kernels, principal_components

__all__ = [
    "loss_surface",
    "plot_bayesian_network",
    "plot_decision_tree",
    "plot_svm_kernels",
    "principal_components",
    "save_loss_path_video",
    "select_top_splits",
]
