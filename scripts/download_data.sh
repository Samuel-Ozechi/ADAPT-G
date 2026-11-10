#!/usr/bin/env bash
# Download IEEE-CIS from Kaggle. You must accept the competition rules and
# configure Kaggle API credentials before running this wrapper.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python "$SCRIPT_DIR/data/download_ieee_cis.py" "$@"