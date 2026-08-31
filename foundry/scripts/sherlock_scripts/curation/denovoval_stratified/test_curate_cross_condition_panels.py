from __future__ import annotations

import csv

from curate_cross_condition_panels import (
    CONDITIONS,
    Candidate,
    candidate_preference_key,
    choose_group_candidates,
    load_rasa_overrides,
    parse_optional_int,
)


GROUP = "AHR_len150"


def test_integral_float_rank_is_accepted() -> None:
    assert parse_optional_int("6.0") == 6


def candidate(
    condition: str,
    name: str,
    *,
    panel: str | None = None,
    slot: int | None = None,
    eligible: bool = True,
    preference_order: int = 1,
    candidate_origin: str = "baseline",
) -> Candidate:
    return Candidate(
        condition=condition,
        group=GROUP,
        ccd="AHR",
        length=150,
        staged_id=name,
        joint_id=f"{condition}__{name}",
        source_path=f"/{name}.cif.gz",
        category="organic",
        ligand_class="small_molecule",
        status="ok",
        rasa_value=0.1 if eligible else 0.2,
        eligible=eligible,
        audit_selection_rank=None,
        continuation_rank=None,
        current_panel=panel,
        current_slot=slot,
        candidate_origin=candidate_origin,
        preference_order=preference_order,
    )


def test_extra_candidate_follows_every_unranked_baseline_candidate() -> None:
    extra = candidate(
        CONDITIONS[0],
        "000_extra",
        candidate_origin="seed8_n100",
    )
    baseline = candidate(CONDITIONS[0], "zzz_baseline")

    assert candidate_preference_key(baseline) < candidate_preference_key(extra)


def test_rasa_override_preserves_candidate_namespace(tmp_path) -> None:
    source_path = tmp_path / "raw.cif.gz"
    json_path = tmp_path / "raw.json"
    source_path.write_bytes(b"cif")
    json_path.write_text("{}")
    manifest_path = tmp_path / "input.tsv"
    result_path = tmp_path / "result.tsv"
    rasa_id = f"{CONDITIONS[0]}__seed8_n100__raw"
    manifest_row = {
        "staged_id": rasa_id,
        "condition": CONDITIONS[0],
        "group": "GLU_len150",
        "ccd": "GLU",
        "length": "150",
        "source_path": str(source_path),
        "json_path": str(json_path),
        "candidate_staged_id": "seed8_n100__raw",
        "candidate_origin": "seed8_n100",
    }
    result_row = {
        "staged_id": rasa_id,
        "condition": CONDITIONS[0],
        "ccd": "GLU",
        "length": "150",
        "category": "metal_free_ligands",
        "ligand_class": "metal_free_ligand",
        "source_path": str(source_path),
        "json_path": str(json_path),
        "status": "ok",
        "detail": "",
        "n_target_heavy_atoms": "10",
        "n_target_residues": "1",
        "rasa_value": "0.1",
    }
    for path, row in ((manifest_path, manifest_row), (result_path, result_row)):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=tuple(row), delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerow(row)

    overrides = load_rasa_overrides(manifest_path, result_path)

    override = overrides[(CONDITIONS[0], "seed8_n100__raw")]
    assert override["candidate_origin"] == "seed8_n100"
    assert override["group"] == "GLU_len150"
    assert override["rasa_value"] == 0.1


def test_main_retention_wins_a_cross_condition_cluster_conflict() -> None:
    condition0, condition1 = CONDITIONS
    rows: list[Candidate] = []
    assignments: dict[str, str] = {}

    for slot in range(5):
        row = candidate(condition0, f"c0_main_{slot}", panel="main", slot=slot)
        rows.append(row)
        assignments[row.joint_id] = f"c0_main_cluster_{slot}"
    c0_subset = candidate(condition0, "c0_subset", panel="subset", slot=5)
    c0_replacement = candidate(condition0, "c0_replacement", preference_order=20)
    rows.extend((c0_subset, c0_replacement))
    assignments[c0_subset.joint_id] = "shared_cluster"
    assignments[c0_replacement.joint_id] = "c0_replacement_cluster"

    for slot in range(5):
        row = candidate(condition1, f"c1_main_{slot}", panel="main", slot=slot)
        rows.append(row)
        assignments[row.joint_id] = (
            "shared_cluster" if slot == 0 else f"c1_main_cluster_{slot}"
        )
    c1_subset = candidate(condition1, "c1_subset", panel="subset", slot=5)
    c1_replacement = candidate(condition1, "c1_replacement", preference_order=20)
    rows.extend((c1_subset, c1_replacement))
    assignments[c1_subset.joint_id] = "c1_subset_cluster"
    assignments[c1_replacement.joint_id] = "c1_replacement_cluster"

    selected, _ = choose_group_candidates(rows, assignments)

    assert selected is not None
    selected_ids = {row.joint_id for row in selected}
    assert c0_subset.joint_id not in selected_ids
    assert c0_replacement.joint_id in selected_ids
    assert f"{condition1}__c1_main_0" in selected_ids
    assert sum(row.current_panel == "main" for row in selected) == 10
    assert len({assignments[row.joint_id] for row in selected}) == 12


def test_rasa_cutoff_candidate_is_not_selectable() -> None:
    rows: list[Candidate] = []
    assignments: dict[str, str] = {}
    excluded: Candidate | None = None
    replacement: Candidate | None = None
    for condition_index, condition in enumerate(CONDITIONS):
        for slot in range(6):
            eligible = not (condition_index == 0 and slot == 0)
            row = candidate(
                condition,
                f"c{condition_index}_{slot}",
                panel="main" if slot < 5 else "subset",
                slot=slot,
                eligible=eligible,
            )
            rows.append(row)
            assignments[row.joint_id] = f"c{condition_index}_cluster_{slot}"
            if not eligible:
                excluded = row
        if condition_index == 0:
            replacement = candidate(condition, "below_cutoff_replacement", preference_order=30)
            rows.append(replacement)
            assignments[replacement.joint_id] = "replacement_cluster"

    selected, _ = choose_group_candidates(rows, assignments)

    assert selected is not None
    assert excluded is not None and excluded not in selected
    assert replacement is not None and replacement in selected


def test_selection_fails_closed_when_one_condition_has_only_five_clusters() -> None:
    rows: list[Candidate] = []
    assignments: dict[str, str] = {}
    for condition_index, condition in enumerate(CONDITIONS):
        count = 5 if condition_index == 0 else 6
        for slot in range(count):
            row = candidate(condition, f"c{condition_index}_{slot}")
            rows.append(row)
            assignments[row.joint_id] = f"c{condition_index}_cluster_{slot}"

    selected, diagnostics = choose_group_candidates(rows, assignments)

    assert selected is None
    assert diagnostics["eligible_unique_clusters_by_condition"][CONDITIONS[0]] == 5
