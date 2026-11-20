# AGENTS.md

## Setup & install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,data]"
python scripts/data/download_ieee_cis.py   # downloads IEEE-CIS CSVs into data/raw/ieee_cis/
```

The project is named `adapt-g` (package dir: `src/adapt_g/`). Editable install via `pip install -e .` is required — tests and scripts import from the `adapt_g` namespace.

## Core commands

### Run tests

```bash
python -m pytest tests/           # all tests
python -m pytest tests/ -k <name>  # single test
```

No other test runners. Fixture CSVs are at `tests/fixtures/` (referenced in `conftest.py`). Many tests use synthetic data and don't need the real dataset.

### Run training scripts (all require the real IEEE-CIS dataset under `data/raw/ieee_cis/`)

```bash
python scripts/smoke_run.py                                    # plumbing check (no data needed)
python scripts/run_preprocessing.py --config configs/base.yaml --config configs/data/ieee_cis.yaml
python scripts/train_tabular.py    --config configs/base.yaml --config configs/data/ieee_cis.yaml
python scripts/train_temporal.py   --config configs/base.yaml --config configs/data/ieee_cis.yaml
python scripts/build_graph.py      --config configs/base.yaml --config configs/data/ieee_cis.yaml
python scripts/train_gnn.py        --config configs/base.yaml --config configs/data/ieee_cis.yaml
```

Configs are YAML files deep-merged in order: later `--config` values override earlier ones. There is no CLI shorthand — you must pass every config you want merged.

## Architecture

- **`configs/`** — YAML configs deep-merged at runtime. `base.yaml` sets paths/seed/splits; `data/ieee_cis.yaml` sets column names and field groupings; `models/` has skeleton model configs.
- **`src/adapt_g/`** — The installable package. Major subpackages:
  - `data/` — loaders, schema validation, chronological time-based splits
  - `features/` — preprocessing, rolling aggregations, sequence builder, tabular fold builder
  - `models/` — baselines (`MajorityBaseline`, `RuleBaseline`, `LogRegBaseline`, `LightGBMBaseline`), temporal transformer (`TemporalTransformer`), GNN (`GraphSAGE`)
  - `evaluation/` — metric computation, report generation, threshold selection, bootstrap CIs
  - `graph/` — temporal bipartite graph builder and k-hop neighbourhood sampler
  - `config.py` — `load_config()` deep-merges YAML; `config_hash()` produces a SHA256 digest
  - `manifest.py` — every run writes `runs/<run_id>/run_manifest.json` with git commit, config hash, and metadata
  - `seeding.py` — `set_seed()` seeds `random`, `numpy`, env `PYTHONHASHSEED`, and optionally `torch`
- **`scripts/`** — Driver scripts. Each corresponds to an AG ticket (AG-003 through AG-009). They write outputs to `runs/<run_id>/`.

## Important conventions

- **Run IDs** use `new_run_id(tag)` → `"<tag>-<YYYY-MM-DD>-<HHMMSS>-<shorthex>"`. Valid tags: `baseline`, `transformer`, `gnn`, `evaluation`, `unified`, `ablation`, `graph`, `smoke`. Non-valid tags raise `ValueError`.
- **Chronological splits** use quantile boundaries on `TransactionDT` (not random). Defaults: train [0, 0.8), val [0.8, 0.9), test [0.9, 1.0].
- **numpy<2** is pinned in `pyproject.toml`. Many upstream libraries may not yet support numpy 2.
- **Evaluation reports** are written as JSON + Markdown pairs. Report structure: train/val/test metrics + threshold selection + bootstrap CIs.
- **Synthetic fixture fallback**: some scripts fall back to synthetic data when the raw data directory is missing (e.g. `run_preprocessing.py`). Others silently use synthetic if the CSV file isn't found (e.g. `train_tabular.py` — see its `FileNotFoundError` catch missing, it actually relies on the loader raising an error but there is no try/except; it just fails). The `test_train_tabular_driver.py` integration test deliberately points to a missing file and expects synthetic fallback NOT to happen — the script requires real data.
- **Graph construction** uses train ∪ val for edges (not train-only). Test snapshots use a strict-< time cutoff for temporal leakage prevention.
- **`freeze_reference.py`** pins a completed run as a reference in `configs/frozen_references.json`. It refuses to freeze synthetic runs.

## Key gotchas

- Scripts must be run from the repo root. They use relative paths (`configs/`, `data/`, `runs/`).
- The `--config` flag must be repeated for each YAML file. Not `--config a.yaml b.yaml`, but `--config a.yaml --config b.yaml`.
- `pip install -e ".[dev,data]"` installs the test deps plus `gdown` for dataset download. The simpler `pip install -r requirements.txt` also works but only pulls `[data]` (it's the same as `pip install -e .[data]`).
- Test fixtures live in `tests/fixtures/` as CSV files. The `conftest.py` loads them by relative path from that directory.
- There are no CI workflows, pre-commit hooks, or lint/format configs in this repo.