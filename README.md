# RCUF-ECG

Reproducibility code and compact artifacts for a five-label, multilabel PTB-XL selective ECG diagnosis study. The repository implements the complete clean, in-distribution experimental path: data preparation, ResNet1D-small training, validation-based label thresholding, temperature scaling, uncertainty scoring, rank-calibrated uncertainty fusion (RCUF), selective evaluation, paired bootstrap analysis, classwise calibration, and subgroup rejection audits.


## What is included

- `src/ptbxl/`: dataset, diagnostic-superclass mapping, ResNet1D, uncertainty scores, calibration metrics, and selective metrics.
- `scripts/`: executable preparation, training, inference, calibration, uncertainty, evaluation, aggregation, and reproduction commands.
- `configs/`: the five-seed paper protocol.
- `splits/`: exact ECG IDs used for train, validation, calibration, and test.
- `artifacts/checkpoints/`: five trained ResNet1D-small checkpoints.
- `artifacts/predictions/`: per-record calibration/test logits, probabilities, predictions, risk scores, and temperature-scaled predictions.
- `artifacts/calibration/`: label thresholds, temperatures, prototype artifacts, and calibration rank references.
- `results/`: recalculated tables from the archived per-record predictions.

The raw PTB-XL waveform archive is not redistributed. Download PTB-XL v1.0.3 separately and accept the dataset's license and usage terms.

## Environment

The archived training configuration uses Python 3.11 and CUDA. Create the environment with either:

```bash
conda env create -f environment.yml
conda activate rcuf-ecg
```

or:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the PyTorch build appropriate for the local CUDA driver when the generic requirement does not select it automatically. The formal five-seed training configuration requires CUDA; CPU execution is permitted by the code but is not a claim of timing equivalence.

## Verify the reported results without PTB-XL waveforms

The checked-in per-record predictions are sufficient to recompute the manuscript's principal clean-test results:

```bash
python scripts/reproduce_results.py --bootstrap 500
```

Outputs are written to `results/` by default. The command recomputes, rather than copies, the baseline metrics, three-component RCUF results, dense and coarse partial AURC, calibration-only cutoff transfer, paired bootstrap intervals, accepted/rejected reliability, per-class calibration, and age/sex rejection-rate summaries.

Expected five-seed results (mean ± sample standard deviation):

| Metric | Recalculated result |
|---|---:|
| Macro-AUC | 0.8971 ± 0.0018 |
| Macro-F1 | 0.6864 ± 0.0061 |
| Micro-F1 | 0.7334 ± 0.0039 |
| RCUF Macro-F1 at 90% coverage | 0.7101 ± 0.0062 |
| RCUF risk at 90% coverage | 0.2372 ± 0.0033 |
| RCUF coarse pAURC (coverage 0.50–1.00) | 0.1026 ± 0.0017 |
| Rejected/accepted error ratio at 90% coverage | 2.2357 ± 0.2540 |
| Raw Micro-ECE | 0.0474 ± 0.0098 |
| Temperature-scaled Micro-ECE | 0.0311 ± 0.0053 |

The audit passes only when both the mean and standard deviation round to the manuscript values at four decimals.

## Full reproduction from PTB-XL

Expected data layout:

```text
<PTBXL_ROOT>/ptbxl_database.csv
<PTBXL_ROOT>/scp_statements.csv
<PTBXL_ROOT>/records100/...
```

Prepare the 100 Hz tensors and export the split IDs:

```bash
python scripts/prepare_data.py \
  --data-root <PTBXL_ROOT> \
  --out-dir outputs_ptbxl \
  --split-dir splits
```

Run all five seeds and the complete downstream analysis:

```bash
python scripts/run_all.py \
  --config configs/pipeline.yaml \
  --out-dir outputs_ptbxl/p8 \
  --bootstrap 500
```

For a single seed:

```bash
python scripts/run_seed.py \
  --seed 1 \
  --config configs/pipeline.yaml \
  --out-dir outputs_ptbxl/p8
```

## Experimental protocol

- Dataset: PTB-XL v1.0.3, 100 Hz records.
- Task: five diagnostic superclasses in the fixed order `NORM`, `MI`, `STTC`, `CD`, `HYP`.
- Input: 12 leads × 1,000 samples; central cropping or zero padding followed by record-wise, lead-wise median/IQR normalization.
- Split: folds 1–8 train; fold 9 even `ecg_id` validation; fold 9 odd `ecg_id` calibration; fold 10 test.
- Split sizes: 17,418 train; 1,102 validation; 1,081 calibration; 2,198 test.
- Seeds: 1, 2, 3, 4, 5.
- Backbone: ResNet1D-small, base width 32, 256-dimensional embedding, dropout 0.2.
- Optimization: AdamW, learning rate 0.001, weight decay 0.0001, effective batch size 64, up to 60 epochs, validation Macro-AUC early stopping with patience 10.
- Label decisions: per-label thresholds selected on validation data only.
- Calibration: scalar and vector temperature candidates fitted on calibration logits; the calibration NLL rule selects the retained candidate.
- Paper RCUF: mean of calibration empirical ranks for MSP risk, predictive entropy, and negative mean absolute logit magnitude. Lower score is accepted.
- Prototype distance: retained as a separate score and four-component ablation; it is not part of the paper's main RCUF score.
- Test isolation: test data are not used to select label thresholds, temperatures, fusion components, prototypes, or acceptance cutoffs.

`RCUF-TA` in `scripts/reproduce_results.py` is an exploratory threshold-aware extension. The manuscript's main method remains `RCUF_3`.

## Tests

```bash
python -m unittest discover -s tests -v
```

The test suite checks core ranking/selective metrics, repository scope, and a complete prediction-only recalculation with a short bootstrap run.

## Reproducibility boundary

The repository can verify the current numerical tables without the waveform archive because it includes the exact per-record clean calibration/test outputs. Re-training the backbone or regenerating logits requires PTB-XL. Hardware identity and the exact historical CUDA/cuDNN patch versions were not preserved, so bitwise-identical retraining is not guaranteed; the provided checkpoints and prediction-level audit are the authoritative trace for the reported run.
