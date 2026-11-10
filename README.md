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

IEEE-CIS is distributed through the Kaggle competition and is not included in
this repository. Accept the competition rules on Kaggle and configure the
Kaggle API credentials before downloading. Keep credentials out of Git.

```bash
pip install -r requirements.txt
python scripts/data/download_ieee_cis.py
```

The downloader saves `train_transaction.csv` and `train_identity.csv` under
`data/raw/ieee_cis/`. Use `--force` to replace existing copies.

## Google Colab quickstart

Run these commands in a Colab terminal to clone the code and install its
dependencies:

```bash
git clone --depth 1 https://github.com/Samuel-Ozechi/ADAPT-G.git
cd ADAPT-G
pip install -r requirements.txt
```

In Colab, add `KAGGLE_USERNAME` and `KAGGLE_KEY` as notebook secrets, then run
this in a notebook cell to create the Kaggle credentials file in the temporary
runtime home directory (not in the repository):

```python
import json
from pathlib import Path
from google.colab import userdata

kaggle_dir = Path.home() / ".kaggle"
kaggle_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
credentials = {
	"username": userdata.get("KAGGLE_USERNAME"),
	"key": userdata.get("KAGGLE_KEY"),
}
credential_file = kaggle_dir / "kaggle.json"
credential_file.write_text(json.dumps(credentials))
credential_file.chmod(0o600)
```

Then, in the Colab terminal:

```bash
python scripts/data/download_ieee_cis.py
python scripts/train_tabular.py --config configs/base.yaml --config configs/data/ieee_cis.yaml
```

The Colab runtime's local files are temporary; copy the dataset or run outputs
to mounted Google Drive if they need to persist after the runtime ends.