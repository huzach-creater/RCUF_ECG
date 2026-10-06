# RCUF-ECG

Code and experimental artifacts for RCUF-based selective diagnosis on PTB-XL.

## Repository structure

- `src/ptbxl/`: data processing, ResNet1D, uncertainty scores, and evaluation metrics
- `scripts/`: training and evaluation scripts
- `configs/`: experiment configurations
- `splits/`: ECG IDs for the train, validation, calibration, and test sets
- `artifacts/`: trained checkpoints, calibration parameters, and saved predictions
- `results/`: numerical results reported in the paper

## Installation

```bash
conda env create -f environment.yml
conda activate rcuf-ecg
```

Alternatively:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the reported results

The main results can be recalculated from the saved predictions without downloading the ECG waveforms:

```bash
python scripts/reproduce_results.py --bootstrap 500
```

The output tables are written to `results/`. Calibration summaries use the validation-threshold operational decisions; event-probability ECE is reported separately.

## Training from scratch

Download PTB-XL v1.0.3 and arrange the files as follows:

```text
<PTBXL_ROOT>/ptbxl_database.csv
<PTBXL_ROOT>/scp_statements.csv
<PTBXL_ROOT>/records100/...
```

Prepare the data:

```bash
python scripts/prepare_data.py \
  --data-root <PTBXL_ROOT> \
  --out-dir outputs_ptbxl \
  --split-dir splits
```

Run the five-seed experiment:

```bash
python scripts/run_all.py \
  --config configs/pipeline.yaml \
  --out-dir outputs_ptbxl/p8 \
  --bootstrap 500
```

To run one seed only:

```bash
python scripts/run_seed.py \
  --seed 1 \
  --config configs/pipeline.yaml \
  --out-dir outputs_ptbxl/p8
```

## Tests

```bash
python -m unittest discover -s tests -v
```
