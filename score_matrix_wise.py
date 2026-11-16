"""Evaluation entry point for matrix-wise weighting."""

from nsc import original_nsc
from pb_nsc_score import matrix_wise_score


def score(config):
    return matrix_wise_score(config)


def ablation(config):
    return original_nsc(config)