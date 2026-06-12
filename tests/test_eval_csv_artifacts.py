from __future__ import annotations

import importlib.util
import tarfile
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "sherlock_scripts"
    / "jinho"
    / "utils"
    / "eval_csv_artifacts.py"
)
SPEC = importlib.util.spec_from_file_location("eval_csv_artifacts", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
eval_csv_artifacts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(eval_csv_artifacts)


def write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "sample_id,value\n" + "".join(f"{sample_id},{value}\n" for sample_id, value in rows)
    path.write_text(text, encoding="utf-8")


def test_merge_suffix_array_shards(tmp_path: Path) -> None:
    step_dir = tmp_path / "experiment" / "step_100"
    write_csv(step_dir / "seq_recovery_metrics_array_0.csv", [("a", "1")])
    write_csv(step_dir / "seq_recovery_metrics_array_1.csv", [("b", "2")])

    ok = eval_csv_artifacts.merge_artifacts([tmp_path / "experiment"])

    assert ok
    assert (step_dir / "seq_recovery_metrics.csv").read_text(encoding="utf-8") == (
        "sample_id,value\n"
        "a,1\n"
        "b,2\n"
    )


def test_merge_array_directory_shards_without_merging_sweep_dirs(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    write_csv(root / "array_0" / "metrics" / "foo.csv", [("a", "1")])
    write_csv(root / "array_1" / "metrics" / "foo.csv", [("b", "2")])
    write_csv(root / "scale_0p1_recycles1" / "metrics" / "foo.csv", [("scale", "9")])

    ok = eval_csv_artifacts.merge_artifacts([root])

    assert ok
    assert (root / "merged" / "metrics" / "foo.csv").read_text(encoding="utf-8") == (
        "sample_id,value\n"
        "a,1\n"
        "b,2\n"
    )
    assert (root / "scale_0p1_recycles1" / "metrics" / "foo.csv").read_text(
        encoding="utf-8"
    ) == "sample_id,value\nscale,9\n"


def test_gather_preserves_relative_paths_and_excludes_raw_shards_by_default(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    write_csv(root / "scale_0p1_recycles1" / "metrics" / "foo.csv", [("scale", "9")])
    write_csv(root / "manifests" / "run_status.csv", [("status", "1")])
    write_csv(root / "array_0" / "metrics" / "foo.csv", [("raw", "0")])
    write_csv(root / "step_100" / "bar_array_0.csv", [("raw_suffix", "0")])
    output_tar = tmp_path / "collected" / "out.tar.gz"

    copied = eval_csv_artifacts.gather_artifacts(
        [root],
        output_tar,
        include_array_shards=False,
    )

    assert copied == 2
    with tarfile.open(output_tar, "r:gz") as tar:
        names = sorted(tar.getnames())
    assert names == [
        "experiment/manifests/run_status.csv",
        "experiment/scale_0p1_recycles1/metrics/foo.csv",
    ]
