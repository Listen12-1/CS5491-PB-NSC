# Path-Balanced Neural Spectral Capacity

CS5491 Artificial Intelligence course project, Group G047.

This project studies whether projection-shape balance weighting improves
architecture ranking and fixed-budget selection beyond original NSC.

## Environment

- Python 3.9 or later; tested locally with Python 3.13.12.
- Python standard library only.
- No GPU, model training, model download or LLM API is required.

Recorded local environment:
Windows 10, AMD Ryzen 5 5500U CPU, 16GB RAM.

Run all commands from this directory.
Use `python3` instead of `python` if required by your system.

## Source and contribution

The course starter supplies the MP capacity calculation, original NSC,
positional example, archived data and evaluation harness.
See README.md and data/provenance.json for its conventions and limitations.

Our contribution is the PB-NSC weighting heuristic, targeted ablations,
numerical checks and development analyses.

The score models projection-shape balance, not layer ordering, complete
graph topology or exact end-to-end information capacity.

## Correctness checks

```sh
python check_starter.py
python check_pb_nsc.py
```

Expected output: PASS messages.
Neither script evaluates final performance. The starter check verifies
final-label file integrity by hash without parsing its outcomes.

## Development evaluation

Original starter:

```sh
python evaluate.py --split development --output baseline_development.json
```

Full PB-NSC:

```sh
python evaluate.py --split development --student pb_nsc_score.py --output pb_nsc_development.json
```

Attention-only:

```sh
python evaluate.py --split development --student score_attention_only.py --output attention_only_development.json
```

FFN-only:

```sh
python evaluate.py --split development --student score_ffn_only.py --output ffn_only_development.json
```

Matrix-wise:

```sh
python evaluate.py --split development --student score_matrix_wise.py --output matrix_wise_development.json
```

In each evaluation output, `student` denotes the selected implementation.
Its `ablation` returns original NSC with all new weighting disabled.

Each evaluation reports ranking, ties, resource-controlled comparisons,
and selection under 10M, 20M, 30M and 50M parameter caps with k=1 and k=3.
Uniform random selection is included by the supplied evaluator.

## Additional development analyses

```sh
python analyze_width.py
python analyze_pair_changes.py
python bootstrap_development.py
python benchmark_scores.py
```

Outputs:

- width_analysis_development.json
- pair_changes_development.csv
- pair_changes_summary_development.json
- bootstrap_development_results.json
- benchmark_development_results.json

The bootstrap uses 2,000 paired architecture resamples within hidden-width
strata, with seed 5491. Resource-controlled pairs are rebuilt in each resample.
Intervals describe architecture-sampling variation conditional on this archive,
not training-seed uncertainty.

Timing uses seven fresh-process runs per method, excluding imports and
data loading while allowing cache reuse within each scoring batch.

Commands overwrite their named output files. Back up recorded results
before rerunning if the originals should be preserved.

## Recorded development findings

- PB-NSC: Spearman rho = 0.975818, Kendall tau-b = 0.880404.
- Original NSC: Spearman rho = 0.969745, Kendall tau-b = 0.861414.
- Resource-controlled tau-b: 0.633333 versus 0.622222.
- Best-selected perplexity improves in two of eight budget/k settings
  and is unchanged in the other six.
- The bootstrap interval for overall Kendall improvement is above zero.
  Intervals for Spearman and resource-controlled improvement include zero.
- No clear advantage over matrix-wise weighting is established.
- All five scoring methods take approximately 0.5 seconds per batch of
  100 development architectures on the recorded laptop.

See the development result JSON and CSV files in this repository for details.

## Approximate runtime

On the recorded laptop:
- Evaluation and descriptive analyses: generally seconds per command.
- 2,000 bootstrap resamples: approximately 14 seconds.
- Timing benchmark: 35 scoring batches plus process overhead.

Runtime varies by machine. Timing fields need not reproduce exactly.

## Final evaluation status

Final performance has not yet been evaluated in this workflow.
The current analysis scripts are development-only.

Freeze the method, evaluation choices and code revision before evaluating
the final split. Do not tune the method using final outcomes.

## References

- Zhu, Zhao and Lu. Neural Spectral Capacity: Measuring and Designing
  Architectures from Network Specification Alone.
  https://arxiv.org/abs/2609.23087
- Javaheripi et al. LiteTransformerSearch: Training-free Neural Architecture
  Search for Efficient Language Models. NeurIPS 2022.
  https://arxiv.org/abs/2203.02094
- CS5491 course starter and topic handout; see README.md and
  data/provenance.json.
