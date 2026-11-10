"""Download the shared IEEE-CIS CSVs from Google Drive."""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import gdown

FILES = (
    "train_transaction.csv",
    "train_identity.csv",
    "test_transaction.csv",
    "test_identity.csv",
)
DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/"
    "1YT3_HhW66pF51OwKlikQiswseV7Y_e1n?usp=sharing"
)
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
        download_dir = Path(temp_dir) / "drive-download"
        download_dir.mkdir()
        gdown.download_folder(
            url=DRIVE_FOLDER_URL,
            output=str(download_dir),
            quiet=False,
            use_cookies=False,
        )

        members_by_name: dict[str, list[Path]] = {name: [] for name in FILES}
        for path in download_dir.rglob("*"):
            if path.is_file() and path.name in members_by_name:
                members_by_name[path.name].append(path)

        missing = [name for name, paths in members_by_name.items() if len(paths) != 1]
        if missing:
            raise RuntimeError(
                "The shared Google Drive folder must contain exactly one copy "
                "of each required CSV; missing or duplicated: "
                f"{', '.join(missing)}. Check the folder's link-sharing access."
            )

        for name in targets:
            shutil.copy2(members_by_name[name][0], output_dir / name)

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

    download(args.output_dir, force=args.force)


if __name__ == "__main__":
    main()
