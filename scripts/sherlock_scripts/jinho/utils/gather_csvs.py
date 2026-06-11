"""
Gather CSV metric files from evaluation step directories and compress into a tar.gz.

Usage:
    python gather_csvs.py <output_tar_gz> <src_dir1> [src_dir2 ...]
    python gather_csvs.py --array-jobs <output_tar_gz> <src_dir1> [src_dir2 ...]
    python gather_csvs.py --recursive --array-jobs <output_tar_gz> <src_dir1> [src_dir2 ...]
"""
import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path

CSV_STEMS = [
    "all_docking_metrics_per_designed_sample",
    "all_sc_metrics_per_designed_sample",
    "tc_all_docking_metrics_per_designed_sample",
    "tc_all_sc_metrics_per_designed_sample",
    "seq_recovery_metrics",
]


def _find_csvs(step_dir: Path, array_jobs: bool) -> list[Path]:
    """Return list of CSV paths to copy from a step directory."""
    found = []
    for stem in CSV_STEMS:
        # Always check the base file
        base = step_dir / f"{stem}.csv"
        if base.exists():
            found.append(base)
        # Additionally check array files if requested
        if array_jobs:
            found.extend(sorted(step_dir.glob(f"{stem}_array_*.csv")))
    return found


def _step_dirs(src_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        return sorted([d for d in src_dir.rglob("step_*") if d.is_dir()])
    return sorted([d for d in src_dir.iterdir() if d.is_dir() and d.name.startswith("step_")])


def _copy_csvs(csvs: list[Path], dest: Path) -> int:
    if not csvs:
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    for csv_path in csvs:
        shutil.copy2(csv_path, dest / csv_path.name)
    return len(csvs)


def gather(
    src_dirs: list[Path],
    output_tar: Path,
    array_jobs: bool = False,
    recursive: bool = False,
):
    total_copied = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        for src_dir in src_dirs:
            src_dir = src_dir.resolve()
            if not src_dir.is_dir():
                print(f"  [SKIP] Not a directory: {src_dir}")
                continue

            exp_name = src_dir.name
            print(f"  Processing: {exp_name}")

            step_dirs = _step_dirs(src_dir, recursive=recursive)

            if step_dirs:
                # Case 1: step_* subdirectories exist, directly or recursively.
                for step_dir in step_dirs:
                    dest = tmp / exp_name / step_dir.relative_to(src_dir)
                    csvs = _find_csvs(step_dir, array_jobs)
                    if csvs:
                        total_copied += _copy_csvs(csvs, dest)
                    else:
                        print(f"    [SKIP] No CSVs in {step_dir.relative_to(src_dir)}")
                print(f"    {len(step_dirs)} step dirs processed")
            else:
                # Case 2: CSVs directly in src_dir (no step_* subdirectories)
                csvs = _find_csvs(src_dir, array_jobs)
                if csvs:
                    dest = tmp / exp_name
                    total_copied += _copy_csvs(csvs, dest)
                    print(f"    {len(csvs)} CSVs found directly in directory")
                else:
                    print(f"    No step_* directories or CSVs found")

        # Create tar.gz
        output_tar = output_tar.resolve()
        output_tar.parent.mkdir(parents=True, exist_ok=True)

        with tarfile.open(output_tar, "w:gz") as tar:
            for item in tmp.iterdir():
                tar.add(item, arcname=item.name)

    size_mb = output_tar.stat().st_size / (1024 * 1024)
    print(f"\nDone: {total_copied} CSV files -> {output_tar} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Gather CSV metrics into a tar.gz")
    parser.add_argument("output_tar", type=Path, help="Output .tar.gz path")
    parser.add_argument("src_dirs", type=Path, nargs="+", help="Source eval directories")
    parser.add_argument("--array-jobs", action="store_true",
                        help="Also collect *_array_N.csv files from array jobs")
    parser.add_argument("--recursive", action="store_true",
                        help="Find step_* directories recursively, including two-stage outputs")
    args = parser.parse_args()
    gather(args.src_dirs, args.output_tar, array_jobs=args.array_jobs, recursive=args.recursive)


if __name__ == "__main__":
    main()
