# Adapt-G

Adversarial Diffusion and Probabilistic Transformer-Graphs for Financial Fraud Detection.

Research project. See `docs/research_spec.md` and `docs/threat_model.md`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Dataset download

The four IEEE-CIS CSVs are available in our publicly shared Google Drive
folder. No Kaggle credentials are needed for this download.

```bash
pip install -r requirements.txt
python scripts/data/download_ieee_cis.py
```

The downloader saves the train and test transaction/identity CSVs under
`data/raw/ieee_cis/`. It only downloads missing files by default, preserving
files already there; use `--force` to replace them. The test data has no
`isFraud` labels, so it is for prediction/submission rather than labeled test
evaluation.

## Google Colab quickstart

Run these commands in a Colab terminal to clone the code and install its
dependencies:

```bash
git clone --depth 1 https://github.com/Samuel-Ozechi/ADAPT-G.git
cd ADAPT-G
pip install -r requirements.txt
```

Then, in the Colab terminal, download the dataset directly into the expected
project folder and start training:

```bash
python scripts/data/download_ieee_cis.py
python scripts/train_tabular.py --config configs/base.yaml --config configs/data/ieee_cis.yaml
```

The Colab runtime's local files are temporary; copy the dataset or run outputs
to mounted Google Drive if they need to persist after the runtime ends.