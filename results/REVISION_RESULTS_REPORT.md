# RCUF-ECG reproduction and revision results

## Reproduction verdict

The archived predictions reproduce the manuscript baseline, calibration, and paper-defined three-component RCUF results.

| Metric | Paper mean | Recalculated mean | Paper SD | Recalculated SD | Match |
|---|---:|---:|---:|---:|---|
| Macro-AUC | 0.897100 | 0.897118 | 0.001800 | 0.001778 | True |
| Macro-F1 | 0.686400 | 0.686400 | 0.006100 | 0.006117 | True |
| Micro-F1 | 0.733400 | 0.733392 | 0.003900 | 0.003853 | True |
| RCUF Macro-F1@90 | 0.710100 | 0.710063 | 0.006200 | 0.006215 | True |
| RCUF Risk@90 | 0.237200 | 0.237166 | 0.003300 | 0.003321 | True |
| RCUF pAURC (coarse) | 0.102600 | 0.102615 | 0.001700 | 0.001683 | True |
| RCUF rejected/accepted ratio | 2.235700 | 2.235683 | 0.254000 | 0.254007 | True |
| Raw operational-decision Micro-ECE | 0.059500 | 0.059488 | 0.011300 | 0.011318 | True |
| Temperature-scaled operational-decision Micro-ECE | 0.047500 | 0.047532 | 0.008900 | 0.008934 | True |
| Raw operational-decision mean class-wise ECE | 0.070600 | 0.070600 | 0.007700 | 0.007723 | True |
| Temperature-scaled operational-decision mean class-wise ECE | 0.062100 | 0.062072 | 0.006900 | 0.006885 | True |

## Test-set selective results

The revised RCUF-TA score averages the paper RCUF-3 empirical-rank score with a calibration-ranked measure of proximity to the five validation-selected decision thresholds. Lower risk is accepted.

| Method | Macro-F1@90 | Risk@90 | Coarse pAURC | Dense pAURC | Error ratio@90 |
|---|---:|---:|---:|---:|---:|
| MSP | 0.709418 ± 0.005332 | 0.238143 ± 0.003171 | 0.103478 ± 0.001830 | 0.103365 ± 0.001843 | 2.189995 ± 0.278183 |
| Entropy | 0.711055 ± 0.006201 | 0.236010 ± 0.002616 | 0.102790 ± 0.001706 | 0.102615 ± 0.001815 | 2.289438 ± 0.198177 |
| Logit magnitude | 0.710554 ± 0.006337 | 0.236536 ± 0.002933 | 0.102834 ± 0.001802 | 0.102768 ± 0.001763 | 2.264740 ± 0.214347 |
| RCUF-3 (paper) | 0.710063 ± 0.006215 | 0.237166 ± 0.003321 | 0.102615 ± 0.001683 | 0.102545 ± 0.001850 | 2.235683 ± 0.254007 |
| Threshold proximity | 0.704964 ± 0.005213 | 0.239436 ± 0.000684 | 0.103773 ± 0.000627 | 0.103799 ± 0.000774 | 2.127750 ± 0.208941 |
| RCUF-TA (revision) | 0.711025 ± 0.005142 | 0.235152 ± 0.002596 | 0.101516 ± 0.001729 | 0.101435 ± 0.001864 | 2.331225 ± 0.239863 |

## Calibration-set development check

RCUF-TA was specified from the existing thresholding design and checked on the calibration split before reporting test performance.

| Method | Macro-F1@90 | Risk@90 | Coarse pAURC |
|---|---:|---:|---:|
| RCUF-3 (paper) | 0.696344 ± 0.004047 | 0.239234 ± 0.003848 | 0.105440 ± 0.001609 |
| RCUF-TA (revision) | 0.697054 ± 0.004127 | 0.237615 ± 0.004946 | 0.104849 ± 0.002145 |

## Calibration-derived threshold transfer

The risk cutoff is estimated only on the calibration split and then applied unchanged to clean test data.

| Method | Test coverage | Rejection rate | Accepted Macro-F1 | Accepted risk | Error ratio |
|---|---:|---:|---:|---:|---:|
| RCUF-3 (paper) | 0.891174 ± 0.008665 | 0.108826 ± 0.008665 | 0.712181 ± 0.007466 | 0.234684 ± 0.004422 | 2.252288 ± 0.243837 |
| RCUF-TA (revision) | 0.896997 ± 0.005578 | 0.103003 ± 0.005578 | 0.711911 ± 0.005664 | 0.234156 ± 0.002715 | 2.340296 ± 0.235866 |

## Paired bootstrap comparisons

Intervals use the same aligned-record resample for all five models and average the paired seed-wise differences. They are conditional on these five trained models and do not replace external validation.

| Comparison | Metric | Difference | 95% bootstrap CI | Direction favoring revision |
|---|---|---:|---:|---|
| RCUF-TA (revision) minus RCUF-3 (paper) | macro_f1_90 | 0.000962 | [-0.001168, 0.002775] | positive |
| RCUF-TA (revision) minus RCUF-3 (paper) | risk_90 | -0.002014 | [-0.003814, 0.000249] | negative |
| RCUF-TA (revision) minus RCUF-3 (paper) | dense_paurc | -0.001110 | [-0.001989, -0.000086] | negative |
| RCUF-TA (revision) minus Entropy | macro_f1_90 | -0.000030 | [-0.001848, 0.002234] | positive |
| RCUF-TA (revision) minus Entropy | risk_90 | -0.000858 | [-0.003449, 0.001045] | negative |
| RCUF-TA (revision) minus Entropy | dense_paurc | -0.001179 | [-0.002096, -0.000191] | negative |

## Class-wise event-probability calibration

This event-probability ECE is different from the manuscript's operational-decision ECE. It compares event probabilities directly with binary labels and exposes label-level behavior hidden by the aggregate decision metric.

| Label | Raw ECE | Temperature-scaled ECE |
|---|---:|---:|
| NORM | 0.060561 ± 0.013410 | 0.046645 ± 0.019434 |
| MI | 0.079101 ± 0.018836 | 0.089330 ± 0.019737 |
| STTC | 0.092162 ± 0.018721 | 0.097342 ± 0.022290 |
| CD | 0.094586 ± 0.025255 | 0.112902 ± 0.026914 |
| HYP | 0.196485 ± 0.028384 | 0.210525 ± 0.029382 |

The unweighted mean event-probability ECE across labels changes from 0.104579 to 0.111349. This is a separate diagnostic and does not contradict the improvement in operational-decision Micro-ECE or mean class-wise ECE.

## Subgroup rejection-rate check

Rejection rates below are at global 90% coverage. They are descriptive because age/sex were not used for stratified model selection.

| Method | Group | Rejection rate |
|---|---|---:|
| RCUF-3 (paper) | age=40-64 | 0.087991 ± 0.006398 |
| RCUF-3 (paper) | age=<40 | 0.061268 ± 0.012598 |
| RCUF-3 (paper) | age=65+ | 0.120611 ± 0.004455 |
| RCUF-3 (paper) | sex=0 | 0.088693 ± 0.005395 |
| RCUF-3 (paper) | sex=1 | 0.112195 ± 0.005729 |
| RCUF-TA (revision) | age=40-64 | 0.085450 ± 0.011518 |
| RCUF-TA (revision) | age=<40 | 0.052113 ± 0.013500 |
| RCUF-TA (revision) | age=65+ | 0.125191 ± 0.011258 |
| RCUF-TA (revision) | sex=0 | 0.091696 ± 0.007737 |
| RCUF-TA (revision) | sex=1 | 0.109006 ± 0.008216 |

## Interpretation

RCUF-TA improves the mean clean-test selective risk and pAURC relative to the paper RCUF-3, but it is an exploratory extension on the same dataset. The bootstrap table and subgroup/per-class files should be reported, and external validation is still required before making a superiority claim.
