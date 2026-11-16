"""Evaluation entry point for FFN-only weighting."""

from nsc import original_nsc
from pb_nsc_score import ffn_only_score


def score(config):
    return ffn_only_score(config)


def ablation(config):
    return original_nsc(config)