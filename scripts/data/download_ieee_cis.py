"""Download the IEEE-CIS Fraud Detection competition files from Kaggle.

Before running, accept the competition rules on Kaggle and configure the Kaggle
API credentials. The train and test CSVs are written to data/raw/ieee_cis by default.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

FILES = (
    "train_transaction.csv",
    "train_identity.csv",
    "test_transaction.csv",
    "test_identity.csv",
)
COMPETITION = "ieee-fraud-detection"
REPO_ROOT = Path(__file__).resolve().parents[2]


def download(output_dir: Path, *, force: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = FILES if force else tuple(
        name for name in FILES if not (output_dir / name).exists()
    )
    if not targets:
        print(f"All IEEE-CIS CSVs already exist in {output_dir.resolve()} (use --force to replace).")
        return

    with tempfile.TemporaryDirectory(prefix="adapt-g-ieee-cis-") as temp_dir:
        temp_path = Path(temp_dir)
        archive_dir = temp_path / "download"
        archive_dir.mkdir()
        subprocess.run(
            [
                "kaggle",
                "competitions",
                "download",
                "-c",
                COMPETITION,
                "-p",
                str(archive_dir),
            ],
            check=True,
        )

        archives = list(archive_dir.glob("*.zip"))
        if len(archives) != 1:
            raise RuntimeError(
                f"Expected one Kaggle ZIP archive in {archive_dir}, found {len(archives)}."
            )

        staged_dir = temp_path / "staged"
        staged_dir.mkdir()
        with zipfile.ZipFile(archives[0]) as archive:
            members_by_name: dict[str, list[str]] = {name: [] for name in FILES}
            for member in archive.namelist():
                basename = Path(member).name
                if basename in members_by_name:
                    members_by_name[basename].append(member)

            missing = [name for name, members in members_by_name.items() if len(members) != 1]
            if missing:
                raise RuntimeError(
                    "Kaggle archive must contain exactly one copy of each required file; "
                    f"missing or duplicated: {', '.join(missing)}."
                )

            for name in targets:
                with archive.open(members_by_name[name][0]) as source:
                    with (staged_dir / name).open("wb") as destination:
                        shutil.copyfileobj(source, destination)

        for name in targets:
            (staged_dir / name).replace(output_dir / name)

    print(f"Downloaded {', '.join(targets)} to {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "ieee_cis",
        help="Destination directory (default: data/raw/ieee_cis)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace any existing IEEE-CIS train/test CSVs.",
    )
    args = parser.parse_args()

    try:
        download(args.output_dir, force=args.force)
    except FileNotFoundError as error:
        if error.filename == "kaggle":
            parser.error("Kaggle CLI not found; install this project's requirements.txt first.")
        raise
    except subprocess.CalledProcessError as error:
        parser.error(
            "Kaggle download failed. Confirm API credentials are configured and that "
            "you accepted the IEEE-CIS competition rules on Kaggle. "
            f"(exit code {error.returncode})"
        )


if __name__ == "__main__":
    main()
