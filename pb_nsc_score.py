"""Path-Balanced NSC: a projection-shape balance heuristic."""

import math

from nsc import layers, original_nsc, psi_mp


def balance(m, n):
    """Dimensionless shape balance; positive dimensions are required."""
    if not (
        math.isfinite(m)
        and math.isfinite(n)
        and m > 0
        and n > 0
    ):
        raise ValueError("Dimensions must be positive and finite.")

    return 4 * math.sqrt(m * n) / (math.sqrt(m) + math.sqrt(n)) ** 2


def weighted_score(config, use_balance=True):
    """Compute branch-weighted NSC, or disable all balance weights."""
    d, heads, inner = layers(config)
    values = []

    for h, f in zip(heads, inner):
        # Preserve the starter's grouping and default initialization.
        attention_capacity = (
            3 * h * psi_mp(d, d // h)
            + psi_mp(d, d)
        )
        ffn_capacity = 2 * psi_mp(d, f)

        if use_balance:
            q_attention = math.sqrt(
                balance(d, d // h) * balance(d, d)
            )
            q_ffn = balance(d, f)
        else:
            q_attention = 1.0
            q_ffn = 1.0

        values.append(
            q_attention * attention_capacity
            + q_ffn * ffn_capacity
        )

    return math.fsum(values)


def score(config):
    """Full PB-NSC; higher scores predict better performance."""
    return weighted_score(config, use_balance=True)


def ablation(config):
    """Remove the balance mechanism and return original NSC."""
    return original_nsc(config)


def attention_only_score(config):
    """Apply balance weighting only to the attention branch."""
    d, heads, inner = layers(config)
    values = []

    for h, f in zip(heads, inner):
        attention_capacity = (
            3 * h * psi_mp(d, d // h)
            + psi_mp(d, d)
        )
        ffn_capacity = 2 * psi_mp(d, f)

        q_attention = math.sqrt(
            balance(d, d // h) * balance(d, d)
        )

        values.append(
            q_attention * attention_capacity
            + ffn_capacity
        )

    return math.fsum(values)


def ffn_only_score(config):
    """Apply balance weighting only to the FFN branch."""
    d, heads, inner = layers(config)
    values = []

    for h, f in zip(heads, inner):
        attention_capacity = (
            3 * h * psi_mp(d, d // h)
            + psi_mp(d, d)
        )
        ffn_capacity = 2 * psi_mp(d, f)

        values.append(
            attention_capacity
            + balance(d, f) * ffn_capacity
        )

    return math.fsum(values)



# Matrix-wise 的含义是：每个矩阵使用自己的 b(m,n)，再把容量相加。
# 由于转置对称性，它的 FFN 部分与完整 PB-NSC 相同；主要区别在注意力部分的加权方式。

def matrix_wise_score(config):
    """Weight each projection capacity by its own shape balance."""
    d, heads, inner = layers(config)
    values = []

    for h, f in zip(heads, inner):
        # Q/K/V projections are weighted per head.
        qkv_capacity = (
            3 * h
            * balance(d, d // h)
            * psi_mp(d, d // h)
        )

        # The square attention output projection has balance 1.
        output_capacity = balance(d, d) * psi_mp(d, d)

        # Weight the two FFN projections separately.
        # ffn_capacity = (
        #    balance(d, f) * psi_mp(d, f)
        #    + balance(f, d) * psi_mp(f, d)
        # )
        
        # Reuse the transpose-invariant MP capacity.
        ffn_projection_capacity = psi_mp(d, f)
        ffn_capacity = (
            balance(d, f) + balance(f, d)
        ) * ffn_projection_capacity

        values.append(
            qkv_capacity
            + output_capacity
            + ffn_capacity
        )

    return math.fsum(values)