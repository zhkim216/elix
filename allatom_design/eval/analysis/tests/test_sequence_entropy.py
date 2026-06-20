from pathlib import Path

import math

from allatom_design.eval.analysis.sequence_entropy import (
    MetadataRecord,
    RegionSpec,
    ResidueSequence,
    analyze_records,
    parse_shell_bin,
    sequence_entropy_bits,
)


def _record(example_id: str, sample_id: str) -> MetadataRecord:
    return MetadataRecord(
        metadata_path=Path(f"/tmp/{sample_id}.pt"),
        example_id=example_id,
        designed_sample_id=sample_id,
        designed_sample_seq=None,
        designed_sample_path=Path(f"/tmp/{sample_id}.cif"),
    )


def test_sequence_entropy_bits() -> None:
    assert sequence_entropy_bits("AAAA") == 0.0
    assert sequence_entropy_bits("AC") == 1.0
    assert math.isclose(sequence_entropy_bits("AACC"), 1.0)


def test_parse_shell_bin_accepts_colon_and_dash() -> None:
    assert parse_shell_bin("6:8") == (6.0, 8.0)
    assert parse_shell_bin("8-10") == (8.0, 10.0)


def test_analyze_records_reports_whole_and_region_entropy() -> None:
    records = [_record("input1", "sample0"), _record("input1", "sample1")]
    keys = (("A", 1, "", 0), ("A", 2, "", 1), ("A", 3, "", 2))
    sequences = {
        "sample0": ResidueSequence(
            keys=keys,
            letters=("A", "C", "D"),
            region_masks={
                "all": (True, True, True),
                "pocket_le_8A": (False, True, True),
            },
        ),
        "sample1": ResidueSequence(
            keys=keys,
            letters=("A", "D", "D"),
            region_masks={
                "all": (True, True, True),
                "pocket_le_8A": (False, True, True),
            },
        ),
    }

    result = analyze_records(
        records,
        region_specs=[RegionSpec("all"), RegionSpec("pocket_le_8A", hi=8.0)],
        load_residue_sequence=lambda record: sequences[record.designed_sample_id],
    )

    summary = result.summary_df.set_index(["example_id", "region"])
    assert summary.loc[("input1", "all"), "n_positions"] == 3
    assert summary.loc[("input1", "pocket_le_8A"), "n_positions"] == 2
    assert math.isclose(summary.loc[("input1", "all"), "sum_entropy_bits"], 1.0)
    assert math.isclose(summary.loc[("input1", "pocket_le_8A"), "sum_entropy_bits"], 1.0)
    assert result.failures_df.empty


def test_analyze_records_records_inconsistent_residue_keys_and_continues() -> None:
    records = [_record("input1", "sample0"), _record("input1", "sample1")]
    sequences = {
        "sample0": ResidueSequence(
            keys=(("A", 1, "", 0), ("A", 2, "", 1)),
            letters=("A", "C"),
            region_masks={"all": (True, True)},
        ),
        "sample1": ResidueSequence(
            keys=(("A", 1, "", 0), ("A", 3, "", 2)),
            letters=("A", "D"),
            region_masks={"all": (True, True)},
        ),
    }

    result = analyze_records(
        records,
        region_specs=[RegionSpec("all")],
        load_residue_sequence=lambda record: sequences[record.designed_sample_id],
    )

    assert "residue_keys_do_not_match_group_reference" in set(result.failures_df["reason"])
    summary = result.summary_df.set_index(["example_id", "region"])
    assert summary.loc[("input1", "all"), "n_samples"] == 1
    assert summary.loc[("input1", "all"), "sum_entropy_bits"] == 0.0

