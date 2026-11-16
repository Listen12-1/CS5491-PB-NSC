"""Evaluation entry point for attention-only weighting."""

from nsc import original_nsc
from pb_nsc_score import attention_only_score


def score(config):
    return attention_only_score(config)


def ablation(config):
    return original_nsc(config)