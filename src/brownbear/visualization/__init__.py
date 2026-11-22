"""Figures for the trained models."""

from .bayesian_network import plot_bayesian_network
from .decision_tree import plot_decision_tree, select_top_splits
from .logistic_regression import save_logistic_regression_animation
from .svm import plot_kernels

__all__ = [
    "plot_bayesian_network",
    "plot_decision_tree",
    "plot_kernels",
    "save_logistic_regression_animation",
    "select_top_splits",
]
