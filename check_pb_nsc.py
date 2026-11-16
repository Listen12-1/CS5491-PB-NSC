"""Numerical checks for PB-NSC; no outcome labels are loaded."""

import copy
import json
import math
from pathlib import Path

from nsc import original_nsc
# from pb_nsc_score import ablation, balance, score, weighted_score
from pb_nsc_score import (
    ablation,
    attention_only_score,
    balance,
    ffn_only_score,
    matrix_wise_score,
    score,
    weighted_score,
)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def main():
    # Square matrices.
    for d in [1, 64, 128, 512]:
        assert close(balance(d, d), 1.0)

    # Bounds, transpose symmetry and common-scaling invariance.
    for m, n in [(1, 16), (128, 32), (128, 512), (512, 1601)]:
        value = balance(m, n)
        assert 0 < value <= 1
        assert close(value, balance(n, m))

        for scale in [0.5, 2, 10]:
            assert close(
                value,
                balance(scale * m, scale * n),
            )

    # Known values and increasingly narrow shapes.
    assert close(balance(1, 4), 8 / 9)
    assert close(balance(1, 16), 0.64)

    narrow_values = [
        balance(1, ratio)
        for ratio in [1, 4, 16, 100, 10000]
    ]
    assert all(
        left > right
        for left, right in zip(narrow_values, narrow_values[1:])
    )
    assert balance(1, 10**12) < 0.00001

    # Attention weighting depends on head count, not common width.
    attention_weights = []
    for h in [1, 2, 4, 8, 16]:
        q = math.sqrt(balance(128, 128 // h))
        expected = math.sqrt(
            4 * math.sqrt(h) / (1 + math.sqrt(h)) ** 2
        )
        assert close(q, expected)
        assert close(q, math.sqrt(balance(256, 256 // h)))
        attention_weights.append(q)

    assert all(
        left > right
        for left, right in zip(
            attention_weights, attention_weights[1:]
        )
    )

        # With one head, attention projections are square.
    # Attention-only therefore recovers NSC, and matrix-wise
    # agrees with full PB-NSC.
    one_head_config = {
        "d_model": 128,
        "n_layer": 2,
        "n_head": 1,
        "d_inner": [256, 512],
    }
    assert close(
        attention_only_score(one_head_config),
        original_nsc(one_head_config),
    )
    assert close(
        matrix_wise_score(one_head_config),
        score(one_head_config),
    )

    # With square FFNs, FFN-only recovers NSC.
    square_ffn_config = {
        "d_model": 128,
        "n_layer": 2,
        "n_head": [4, 8],
        "d_inner": 128,
    }
    assert close(
        ffn_only_score(square_ffn_config),
        original_nsc(square_ffn_config),
    )

    # Check each formula independently on a single layer.
    from nsc import psi_mp

    single_layer = {
        "d_model": 128,
        "n_layer": 1,
        "n_head": 4,
        "d_inner": 512,
    }

    qkv = 12 * psi_mp(128, 32)
    output = psi_mp(128, 128)
    ffn = 2 * psi_mp(128, 512)

    b_head = balance(128, 32)
    b_ffn = balance(128, 512)

    assert close(
        attention_only_score(single_layer),
        math.sqrt(b_head) * (qkv + output) + ffn,
    )
    assert close(
        ffn_only_score(single_layer),
        qkv + output + b_ffn * ffn,
    )
    assert close(
        matrix_wise_score(single_layer),
        b_head * qkv + output + b_ffn * ffn,
    )

    print("PASS: attention-only, FFN-only and matrix-wise formulas.")

    # A controlled toy configuration.
    config = {
        "d_model": 128,
        "n_layer": 2,
        "n_head": [4, 4],
        "d_inner": [256, 512],
    }

    baseline = original_nsc(config)
    pb_score = score(config)

    # Check the actual weighted computation with all weights disabled.
    assert close(weighted_score(config, use_balance=False), baseline)
    assert close(ablation(config), baseline)
    assert 0 < pb_score < baseline

    # PB-NSC does not claim layer-order sensitivity.
    reversed_config = copy.deepcopy(config)
    reversed_config["d_inner"].reverse()
    assert close(score(config), score(reversed_config))

    # Square projections: PB-NSC recovers NSC.
    square_config = {
        "d_model": 128,
        "n_layer": 2,
        "n_head": 1,
        "d_inner": 128,
    }
    assert close(score(square_config), original_nsc(square_config))

    # Scalar and equivalent per-layer-list specifications agree.
    list_config = {
        "d_model": 128,
        "n_layer": 2,
        "n_head": [1, 1],
        "d_inner": [128, 128],
    }
    assert close(score(square_config), score(list_config))

    # Validate all archived configurations without loading labels.
    root = Path(__file__).resolve().parent
    configs = json.loads(
        (root / "data" / "architectures.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(configs) == 200

    for archived_config in configs.values():
        original = original_nsc(archived_config)
        value = score(archived_config)

        assert math.isfinite(value)
        assert 0 < value <= original + 1e-10
        assert close(
            weighted_score(archived_config, use_balance=False),
            original,
        )
        assert close(ablation(archived_config), original)

    print("PASS: balance bounds, square, transpose, scaling and narrow shapes.")
    print("PASS: head-count behavior and scalar/list configurations.")
    print("PASS: baseline recovery and 200 configuration checks.")
    print("PASS: FFN reorder invariance; PB-NSC is not order-sensitive.")
    print(f"Toy original NSC: {baseline:.9f}")
    print(f"Toy PB-NSC:       {pb_score:.9f}")


if __name__ == "__main__":
    main()