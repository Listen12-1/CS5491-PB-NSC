# Final Results Summary

| Method | Spearman rho | Kendall tau-b | Controlled tau-b |
|---|---:|---:|---:|
| original_nsc | 0.965941 | 0.843232 | 0.329730 |
| position_example | 0.968977 | 0.853333 | 0.318919 |
| pb_nsc | 0.973969 | 0.865455 | 0.351351 |
| reported_parameters | 0.741854 | 0.589495 | 0.275676 |
| reference_flops | 0.965101 | 0.842424 | 0.329730 |
| attention_only | 0.965473 | 0.841616 | 0.318919 |
| ffn_only | 0.973861 | 0.864646 | 0.351351 |
| matrix_wise | 0.973669 | 0.864242 | 0.340541 |

## Interpretation

PB-NSC improves overall ranking on this fixed archive. Overall Spearman
and Kendall differences versus NSC have positive final bootstrap intervals.
The controlled improvement interval includes zero; geometric aggregation
has no established distinct advantage over matrix-wise weighting.

PB-NSC matches NSC's best-selected PPL in all eight final budget/k settings.
The width-512 controlled subset has negative tau-b for both methods.
Most ranking improvement involves cross-width comparisons; FFN shape
preferences are an important descriptive pattern, not causal evidence.

Development uses validation PPL and final uses test PPL on different
architectures. Do not directly compare absolute PPL across splits.
Bootstrap uncertainty is not training-seed uncertainty.

Complete derived tables are in reports/. Their source is the saved JSON
and they can be regenerated with python prepare_submission.py.
