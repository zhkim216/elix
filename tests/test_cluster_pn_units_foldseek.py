from pathlib import Path

import atomworks.enums as aw_enums
import numpy as np
import pandas as pd

from allatom_design.data.preprocessing.atomworks.cluster_pn_units_foldseek import (
    CLUSTER_ID_COLUMN,
    assign_foldseek_and_hash_clusters,
    build_auth_to_label_chain_lookup,
    build_chain_representative_map,
    build_foldseek_easy_cluster_command,
    build_single_protein_chain_lookup,
    compact_representative_labels,
    parse_foldseek_cluster_tsv,
    parse_foldseek_id,
    resolve_structure_path,
)


def _write_atom_site_cif(
    path: Path,
    chain_pairs: list[tuple[str, str]],
) -> None:
    rows = []
    for atom_id, (label_chain, auth_chain) in enumerate(chain_pairs, start=1):
        rows.append(
            "ATOM "
            f"{atom_id} "
            "N N . MET "
            f"{label_chain} "
            f"{atom_id} 1 ? "
            "0.0 0.0 0.0 1.00 1.00 ? "
            f"1 MET {auth_chain} N 1"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "data_test",
                "#",
                "loop_",
                "_atom_site.group_PDB",
                "_atom_site.id",
                "_atom_site.type_symbol",
                "_atom_site.label_atom_id",
                "_atom_site.label_alt_id",
                "_atom_site.label_comp_id",
                "_atom_site.label_asym_id",
                "_atom_site.label_entity_id",
                "_atom_site.label_seq_id",
                "_atom_site.pdbx_PDB_ins_code",
                "_atom_site.Cartn_x",
                "_atom_site.Cartn_y",
                "_atom_site.Cartn_z",
                "_atom_site.occupancy",
                "_atom_site.B_iso_or_equiv",
                "_atom_site.pdbx_formal_charge",
                "_atom_site.auth_seq_id",
                "_atom_site.auth_comp_id",
                "_atom_site.auth_asym_id",
                "_atom_site.auth_atom_id",
                "_atom_site.pdbx_PDB_model_num",
                *rows,
                "#",
            ]
        )
        + "\n"
    )


def _row(
    pdb_id: str,
    q_pn_unit_iid: str,
    q_pn_unit_type: int,
    sequence: str | None = None,
    non_polymer_res_names: str = "",
) -> dict:
    return {
        "pdb_id": pdb_id,
        "rel_path": f"{pdb_id[:2]}/{pdb_id}.cif.gz",
        "q_pn_unit_iid": q_pn_unit_iid,
        "q_pn_unit_type": q_pn_unit_type,
        "q_pn_unit_processed_entity_canonical_sequence": sequence,
        "q_pn_unit_non_polymer_res_names": non_polymer_res_names,
    }


def test_resolve_structure_path_prefers_gz_and_falls_back_to_cif(tmp_path: Path):
    root = tmp_path / "pdb_mirror"
    direct = root / "9c" / "9c0h.cif.gz"
    direct.parent.mkdir(parents=True)
    direct.write_text("gz placeholder")

    resolved, tried = resolve_structure_path("9c/9c0h.cif.gz", root)
    assert resolved == direct
    assert tried == (str(direct),)

    direct.unlink()
    fallback = root / "9c" / "9c0h.cif"
    fallback.write_text("cif placeholder")

    resolved, tried = resolve_structure_path("9c/9c0h.cif.gz", root)
    assert resolved == fallback
    assert tried == (str(direct), str(fallback))


def test_parse_foldseek_id_handles_chain_and_model_labels():
    assert parse_foldseek_id("9c0h_B") == ("9c0h", "B")
    assert parse_foldseek_id("6c00_MODEL_2_A") == ("6c00", "A")
    assert parse_foldseek_id("9c0h.cif.gz_B") == ("9c0h", "B")
    assert parse_foldseek_id("unparseable") is None


def test_cluster_tsv_parsing_and_deterministic_compaction(tmp_path: Path):
    cluster_tsv = tmp_path / "pn_units_cluster.tsv"
    cluster_tsv.write_text(
        "\n".join(
            [
                "rep_b\tmember_b",
                "rep_a\tmember_a",
                "rep_b\tmember_c",
            ]
        )
        + "\n"
    )

    representative_by_member, conflicts = parse_foldseek_cluster_tsv(cluster_tsv)
    assert conflicts == []
    assert representative_by_member == {
        "member_b": "rep_b",
        "member_a": "rep_a",
        "member_c": "rep_b",
    }

    labels = pd.Series(["foldseek:rep_b", "foldseek:rep_a", "foldseek:rep_b"])
    assert compact_representative_labels(labels) == {
        "foldseek:rep_a": 0,
        "foldseek:rep_b": 1,
    }


def test_foldseek_gpu_command_adds_only_gpu_backend_flag(tmp_path: Path):
    command = build_foldseek_easy_cluster_command(
        tmp_path / "input",
        tmp_path / "clusters" / "pn_units",
        tmp_path / "tmp",
        threads=8,
        use_gpu=True,
    )

    assert command[:2] == ["foldseek", "easy-cluster"]
    assert command[-4:] == ["--threads", "8", "--gpu", "1"]
    assert "--min-seq-id" not in command
    assert "--tmscore-threshold" not in command
    assert "--lddt-threshold" not in command
    assert "--alignment-type" not in command


def test_protein_row_mapping_from_pdb_id_and_chain_prefix():
    representative_by_member = {
        "9c0h_B": "9c0h_B",
        "9c0h_C": "9c0h_B",
    }
    chain_to_representative, conflicts, unparsable, mapping_misses = (
        build_chain_representative_map(representative_by_member)
    )
    assert conflicts == []
    assert unparsable == []
    assert mapping_misses == []

    df = pd.DataFrame(
        [
            _row(
                "9c0h",
                "B_1",
                aw_enums.ChainType.POLYPEPTIDE_L.value,
                sequence="M" * 30,
            ),
            _row(
                "9c0h",
                "C_1",
                aw_enums.ChainType.POLYPEPTIDE_L.value,
                sequence="M" * 31,
            ),
            _row(
                "9c0h",
                "D_1",
                aw_enums.ChainType.POLYPEPTIDE_L.value,
                sequence="M" * 32,
            ),
        ]
    )

    out, diagnostics = assign_foldseek_and_hash_clusters(df, chain_to_representative)
    assert out.loc[0, CLUSTER_ID_COLUMN] == out.loc[1, CLUSTER_ID_COLUMN]
    assert out.loc[0, CLUSTER_ID_COLUMN] >= 0
    assert out.loc[2, CLUSTER_ID_COLUMN] == -1
    assert len(diagnostics["unmapped_protein_rows"]) == 1
    assert out[CLUSTER_ID_COLUMN].dtype == np.int32


def test_bare_foldseek_id_maps_only_for_single_chain_metadata():
    df = pd.DataFrame(
        [
            _row(
                "10af",
                "A_1",
                aw_enums.ChainType.POLYPEPTIDE_L.value,
                sequence="M" * 30,
            ),
            _row(
                "10ad",
                "A_1",
                aw_enums.ChainType.POLYPEPTIDE_L.value,
                sequence="M" * 30,
            ),
            _row(
                "10ad",
                "B_1",
                aw_enums.ChainType.POLYPEPTIDE_L.value,
                sequence="M" * 30,
            ),
        ]
    )
    single_chain_by_pdb = build_single_protein_chain_lookup(
        df.assign(q_pn_unit_is_protein=True)
    )
    chain_to_representative, conflicts, unparsable, mapping_misses = (
        build_chain_representative_map(
            {"10af": "10af", "10ad": "10ad"},
            single_chain_by_pdb=single_chain_by_pdb,
        )
    )

    assert conflicts == []
    assert mapping_misses == []
    assert chain_to_representative[("10af", "A")] == "10af"
    assert "10ad" in unparsable


def test_foldseek_auth_chain_maps_to_atomworks_label_chain(tmp_path: Path):
    pdb_mirror_root = tmp_path / "pdb_mirror"
    _write_atom_site_cif(pdb_mirror_root / "10" / "10ad.cif", [("B", "C")])
    df = pd.DataFrame(
        [
            _row(
                "10ad",
                "B_1",
                aw_enums.ChainType.POLYPEPTIDE_L.value,
                sequence="M" * 30,
            ),
        ]
    )

    auth_to_label, mapping_diagnostics = build_auth_to_label_chain_lookup(
        df.assign(q_pn_unit_is_protein=True),
        pdb_mirror_root,
    )
    chain_to_representative, conflicts, unparsable, mapping_misses = (
        build_chain_representative_map(
            {"10ad_C": "10ad_C"},
            auth_to_label_chain=auth_to_label,
        )
    )
    out, diagnostics = assign_foldseek_and_hash_clusters(df, chain_to_representative)

    assert mapping_diagnostics["read_error_count"] == 0
    assert auth_to_label == {("10ad", "C"): "B"}
    assert chain_to_representative == {("10ad", "B"): "10ad_C"}
    assert conflicts == []
    assert unparsable == []
    assert mapping_misses == []
    assert out.loc[0, CLUSTER_ID_COLUMN] >= 0
    assert diagnostics["unmapped_protein_rows"] == []


def test_ambiguous_auth_to_label_mapping_does_not_guess(tmp_path: Path):
    pdb_mirror_root = tmp_path / "pdb_mirror"
    _write_atom_site_cif(
        pdb_mirror_root / "10" / "10ad.cif",
        [("B", "C"), ("D", "C")],
    )
    df = pd.DataFrame(
        [
            _row(
                "10ad",
                "B_1",
                aw_enums.ChainType.POLYPEPTIDE_L.value,
                sequence="M" * 30,
            ),
        ]
    )

    auth_to_label, mapping_diagnostics = build_auth_to_label_chain_lookup(
        df.assign(q_pn_unit_is_protein=True),
        pdb_mirror_root,
    )
    chain_to_representative, conflicts, unparsable, mapping_misses = (
        build_chain_representative_map(
            {"10ad_C": "10ad_C"},
            auth_to_label_chain=auth_to_label,
            ambiguous_auth_label_chains={("10ad", "C")},
        )
    )
    out, diagnostics = assign_foldseek_and_hash_clusters(df, chain_to_representative)

    assert auth_to_label == {}
    assert mapping_diagnostics["ambiguous_auth_chain_mappings"] == [
        {"pdb_id": "10ad", "auth_asym_id": "C", "label_asym_ids": ["B", "D"]}
    ]
    assert chain_to_representative == {}
    assert conflicts == []
    assert unparsable == []
    assert mapping_misses == [
        {
            "member": "10ad_C",
            "pdb_id": "10ad",
            "auth_asym_id": "C",
            "reason": "ambiguous_auth_chain_label_mapping",
        }
    ]
    assert out.loc[0, CLUSTER_ID_COLUMN] == -1
    assert diagnostics["unmapped_protein_rows"] == [
        {"index": 0, "pdb_id": "10ad", "q_pn_unit_iid": "B_1"}
    ]


def test_non_protein_hash_assignment_matches_existing_grouping_semantics():
    df = pd.DataFrame(
        [
            _row(
                "1aaa",
                "A_1",
                aw_enums.ChainType.POLYPEPTIDE_L.value,
                sequence="ACDEFG",
            ),
            _row(
                "1aab",
                "A_1",
                aw_enums.ChainType.POLYPEPTIDE_L.value,
                sequence="ACDEFG",
            ),
            _row("1aac", "A_1", aw_enums.ChainType.DNA.value, sequence="TAACCC"),
            _row("1aad", "A_1", aw_enums.ChainType.DNA.value, sequence="TAACCC"),
            _row("1aae", "A_1", aw_enums.ChainType.RNA.value, sequence="GUGG"),
            _row(
                "1aaf",
                "A_1",
                aw_enums.ChainType.NON_POLYMER.value,
                non_polymer_res_names="ZN,MG",
            ),
            _row(
                "1aag",
                "A_1",
                aw_enums.ChainType.NON_POLYMER.value,
                non_polymer_res_names="MG,ZN",
            ),
            _row("1aah", "A_1", aw_enums.ChainType.OTHER_POLYMER.value),
        ]
    )

    out, diagnostics = assign_foldseek_and_hash_clusters(df, {})
    assert diagnostics["num_clusters"] == 4
    assert out.loc[0, CLUSTER_ID_COLUMN] == out.loc[1, CLUSTER_ID_COLUMN]
    assert out.loc[2, CLUSTER_ID_COLUMN] == out.loc[3, CLUSTER_ID_COLUMN]
    assert out.loc[5, CLUSTER_ID_COLUMN] == out.loc[6, CLUSTER_ID_COLUMN]
    assert out.loc[7, CLUSTER_ID_COLUMN] == -1
    assert out[CLUSTER_ID_COLUMN].dtype == np.int32
