# Final Evaluation Protocol — PB-NSC

Group: CS5491 G047

## Status

This protocol is prepared after development analysis and before final
performance evaluation.

Final results have not yet been evaluated in this workflow.
The freeze revision will be the Git commit containing this protocol,
the split-aware analysis scripts and updated reproduction instructions.
Record that full commit SHA externally before running final commands.

Milestone revision:
2878ddddd08c90af26f9493608105673d03b8bd2

## Frozen scoring methods

The PB-NSC formula remains unchanged from the milestone implementation.

Balance:
b(m,n) = 4*sqrt(m*n) / (sqrt(m)+sqrt(n))^2

Attention branch weight:
q_att = sqrt(b(d,d/H)*b(d,d))

FFN branch weight:
q_ffn = b(d,f)

PB-NSC sums the weighted attention and FFN capacities across layers.
The MP calculation, initialization standard deviation 0.02 and
projection grouping remain those of the course starter.

Methods:
- Original NSC
- Full PB-NSC
- Attention-only weighting
- FFN-only weighting
- Matrix-wise weighting
- Supplied positional example
- Reported total parameters
- Reference FLOPs
- Uniform random selection

The scorer's ablation disables the new mechanism and returns original NSC.

No score uses architecture IDs, outcome lookups or performance labels.

## Data

Use the course-provided fixed final split and its test perplexity labels.
Do not alter data, split membership or reported total parameters.

Development uses validation PPL; final uses test PPL on different
architectures. Their absolute PPL values are not directly interchangeable.

## Ranking

Report:
- Spearman rho against negative PPL
- Kendall tau-b against negative PPL
- Score and outcome ties
- Resource-controlled pair count and tau-b

Resource-controlled pairs:
- Equal hidden width
- Total-parameter min/max >= 0.90

These comparisons are descriptive resource controls, not causal controls.

## Selection

Parameter caps: 10M, 20M, 30M, 50M.
Candidate counts: k=1 and k=3.

Use identical feasible pools and k across methods.
Report best selected PPL and regret relative to the best feasible candidate.
Retain the starter's 200 fixed-seed selection/tie-handling trials.
Report candidate counts and any infeasible settings or failures.

Selection variability is not training-seed uncertainty.

## Bootstrap

- Unit: architecture
- Paired samples shared across methods
- Sampling with replacement within hidden-width strata
- Preserve each stratum's size in the selected split
- Rebuild resource-controlled pairs in every resample
- Preserve multiplicity of repeated architecture draws
- Exclude comparisons between copies of the same architecture
- Repetitions: 2000
- Random seed: 5491
- Interval: 95% percentile bootstrap

Report differences for:
- PB-NSC minus original NSC
- PB-NSC minus matrix-wise

Metrics:
- Spearman rho
- Overall Kendall tau-b
- Resource-controlled tau-b

Report valid and invalid resample counts.
Do not increase repetitions or change the seed to obtain favorable results.

## Descriptive attribution analyses

Run the same analyses established on development:
- Ranking within each hidden-width group
- Ranking reversals relative to original NSC
- Corrected and worsened comparisons
- Same-width and different-width comparisons
- Resource-controlled comparisons
- Mean head-count and mean FFN expansion-ratio patterns

Report subgroup sizes and retain unsuccessful changes.
Overlapping pairs are not independent observations.
These analyses do not establish causal mechanisms.

## Timing

Retain the recorded development fresh-process benchmark:
seven runs per method, with within-batch MP cache reuse.

Clearly identify it as development timing.
Do not present cached evaluator timings as fair speed comparisons.

## Final commands

Run the following only after recording the freeze commit SHA.

### Main method and all baseline comparisons

python evaluate.py --split final --student pb_nsc_score.py --output pb_nsc_final.json

### Targeted ablations

python evaluate.py --split final --student score_attention_only.py --output attention_only_final.json

python evaluate.py --split final --student score_ffn_only.py --output ffn_only_final.json

python evaluate.py --split final --student score_matrix_wise.py --output matrix_wise_final.json

### Descriptive analyses and uncertainty

python analyze_width.py --split final

python analyze_pair_changes.py --split final

python bootstrap_development.py --split final

The bootstrap script retains its historical filename but supports both splits.

## Interpretation and changes

Report all predefined settings, including negative and inconclusive results.
Do not change scoring formulas, choose a new main method or select
subgroups based on final outcomes.

Any necessary correctness fix must be documented with its reason,
code revision and affected results. It must not be used for performance tuning.
Unexpected analyses must be identified as post hoc.

The archive cannot establish training-seed significance or generalization
to other Transformer families.