"""AG-003 end-to-end smoke run: load config, seed, log, write manifest.

Does NOT require the real dataset. Validates plumbing only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from adapt_g.config import config_hash, load_config
from adapt_g.logging_utils import get_logger
from adapt_g.manifest import new_run_id, write_manifest
from adapt_g.seeding import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", action="append", default=None,
        help="YAML config to load; repeat to deep-merge configs in order",
    )
    parser.add_argument("--extra-config", action="append", default=[])
    args = parser.parse_args()

    config_paths = args.config or ["configs/base.yaml"]
    config_paths.extend(args.extra_config)
    cfg = load_config(*config_paths)
    seed = int(cfg["project"]["seed"])
    runs_dir = Path(cfg["paths"]["runs"])
    run_id = new_run_id()
    run_dir = runs_dir / run_id

    logger = get_logger(
        name="adapt_g.smoke",
        log_file=run_dir / "run.log",
        level=cfg["logging"]["level"],
        json_mode=cfg["logging"]["json"],
    )
    set_seed(seed)
    logger.info("smoke run start", extra={"run_id": run_id, "ticket": "AG-003", "seed": seed})

    manifest_path = write_manifest(
        run_dir=run_dir,
        cfg=cfg,
        dataset_version=None,
        split_boundaries=None,
        extra={"ticket": "AG-003", "config_hash_short": config_hash(cfg)},
    )
    logger.info(f"manifest written: {manifest_path}",
                extra={"run_id": run_id, "event": "manifest_written"})
    logger.info("smoke run end", extra={"run_id": run_id, "ticket": "AG-003"})


if __name__ == "__main__":
    main()