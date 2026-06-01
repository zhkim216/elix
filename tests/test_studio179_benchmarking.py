from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from allatom_design.eval.benchmarking.studio179.prepare_lc_seq_des_inputs import (
    make_artifact_stem,
    parse_args,
    prepare_lc_seq_des_inputs,
    resolve_studio179_paths,
)
from allatom_design.eval.benchmarking.studio179.studio179_ligand_map import (
    build_studio179_manifest,
    filter_studio179_manifest,
    normalize_target_signatures,
    parse_studio179_sample_id,
)
from allatom_design.eval.sampling.lc_seq_des_multi import (
    _apply_studio179_userccd_annotation,
    _load_studio179_userccd_annotation,
)


def _write_metadata_csv(path: Path) -> None:
    pd.DataFrame(
        [
            {"ligand": "heme_b_final_0", "ligand_name": "heme_b", "CCD": "HEM"},
            {"ligand": "Cu_ideal", "ligand_name": "Cu", "CCD": "CU"},
            {"ligand": "pyrroloquinoline_quinone_final_0", "ligand_name": "pyrroloquinoline_quinone", "CCD": "PQQ"},
            {"ligand": "thiamine_diphosphate_final_0", "ligand_name": "thiamine_diphosphate", "CCD": "TPP"},
            {"ligand": "flavin_mononucleotide_final_0", "ligand_name": "flavin_mononucleotide", "CCD": "FMN"},
            {"ligand": "Fe2-S2_ideal", "ligand_name": "Fe2-S2", "CCD": "FES"},
            {"ligand": "biotin_final_0", "ligand_name": "biotin", "CCD": "BTN"},
            {"ligand": "acetyl_coenzyme_a_final_0", "ligand_name": "acetyl_coenzyme_a", "CCD": "ACO"},
            {"ligand": "cisplatin_definition", "ligand_name": "cisplatin", "CCD": "CPT"},
            {"ligand": "PFOA_final_0", "ligand_name": "PFOA", "CCD": None},
            {"ligand": "PFOS_final_0", "ligand_name": "PFOS", "CCD": None},
            {
                "ligand": "heme_b_final_0+thiamine_diphosphate_final_0",
                "ligand_name": "heme_b+thiamine_diphosphate",
                "CCD": None,
            },
            {
                "ligand": "heme_b_final_0+Cu_ideal",
                "ligand_name": "heme_b+Cu",
                "CCD": None,
            },
            {
                "ligand": "pyrroloquinoline_quinone_final_0+Cu_ideal",
                "ligand_name": "pyrroloquinoline_quinone+Cu",
                "CCD": None,
            },
        ]
    ).to_csv(path, index=False)


def _write_userccd_manifest(path: Path) -> None:
    pd.DataFrame(
        [
            {"component_id": "S179002", "ligand": "Cu_ideal", "ligand_name": "Cu", "metadata_ccd": "CU", "status": "ok", "error": ""},
            {"component_id": "S179010", "ligand": "pyrroloquinoline_quinone_final_0", "ligand_name": "pyrroloquinoline_quinone", "metadata_ccd": "PQQ", "status": "ok", "error": ""},
            {"component_id": "S179011", "ligand": "thiamine_diphosphate_final_0", "ligand_name": "thiamine_diphosphate", "metadata_ccd": "TPP", "status": "ok", "error": ""},
            {"component_id": "S179052", "ligand": "flavin_mononucleotide_final_0", "ligand_name": "flavin_mononucleotide", "metadata_ccd": "FMN", "status": "ok", "error": ""},
            {"component_id": "S179060", "ligand": "Fe2-S2_ideal", "ligand_name": "Fe2-S2", "metadata_ccd": "FES", "status": "ok", "error": ""},
            {"component_id": "S179079", "ligand": "heme_b_final_0", "ligand_name": "heme_b", "metadata_ccd": "HEM", "status": "ok", "error": ""},
            {"component_id": "S179100", "ligand": "biotin_final_0", "ligand_name": "biotin", "metadata_ccd": "BTN", "status": "ok", "error": ""},
            {"component_id": "S179101", "ligand": "acetyl_coenzyme_a_final_0", "ligand_name": "acetyl_coenzyme_a", "metadata_ccd": "ACO", "status": "ok", "error": ""},
            {"component_id": "S179102", "ligand": "cisplatin_definition", "ligand_name": "cisplatin", "metadata_ccd": "CPT", "status": "ok", "error": ""},
            {"component_id": "S179103", "ligand": "PFOA_final_0", "ligand_name": "PFOA", "metadata_ccd": "", "status": "ok", "error": ""},
            {"component_id": "S179104", "ligand": "PFOS_final_0", "ligand_name": "PFOS", "metadata_ccd": "", "status": "ok", "error": ""},
        ]
    ).to_csv(path, sep="\t", index=False)


def _write_sample_set(tmp_path: Path) -> tuple[Path, Path, Path]:
    metadata_csv = tmp_path / "all_diversity_results.csv"
    list_path = tmp_path / "studio179_all.txt"
    cif_dir = tmp_path / "cifs"
    sample_ids = [
        "length_150_Cu_sample_0",
        "length_150_Cu_sample_1",
        "length_150_heme_b+TPP_sample_0",
        "length_150_heme_b+TPP_sample_1",
        "length_250_heme_b+TPP_sample_0",
        "length_250_heme_b+TPP_sample_3",
        "length_250_heme_b+TPP_sample_4",
        "length_150_heme_b+Cu_2+_sample_0",
        "length_150_PQQ+Cu_2_sample_0",
        "length_150_FMN+Fe2S2_sample_1",
        "length_150_Biotin+Acyl-CoA_sample_2",
        "length_150_cisplatin_sample_3",
        "length_150_PFOA+PFOS_sample_4",
    ]
    _write_metadata_csv(metadata_csv)
    list_path.write_text("\n".join(sample_ids) + "\n")
    cif_dir.mkdir()
    for sample_id in sample_ids:
        (cif_dir / f"{sample_id}.cif").write_text(f"data_{sample_id}\n")
    return metadata_csv, list_path, cif_dir


def test_parse_studio179_sample_id() -> None:
    parsed = parse_studio179_sample_id("length_250_Biotin+Acyl-CoA_sample_17")
    assert parsed.length == 250
    assert parsed.disco_target == "Biotin+Acyl-CoA"
    assert parsed.sample_index == 17


def test_build_manifest_maps_disco_aliases_to_ccd_codes(tmp_path: Path) -> None:
    metadata_csv, list_path, cif_dir = _write_sample_set(tmp_path)

    manifest = build_studio179_manifest(
        sample_id_list=list_path,
        metadata_csv=metadata_csv,
        cif_dir=cif_dir,
    )
    by_target = dict(zip(manifest["disco_target"], manifest["ccd_codes"]))
    by_signature = dict(zip(manifest["disco_target"], manifest["target_signature"]))

    assert by_target["Cu"] == "CU"
    assert by_target["heme_b+TPP"] == "HEM;TPP"
    assert by_target["heme_b+Cu_2+"] == "HEM;CU"
    assert by_target["PQQ+Cu_2"] == "PQQ;CU"
    assert by_target["FMN+Fe2S2"] == "FMN;FES"
    assert by_target["Biotin+Acyl-CoA"] == "BTN;ACO"
    assert by_target["cisplatin"] == "CPT"
    assert by_target["PFOA+PFOS"] == ""
    assert by_signature["Cu"] == "CU"
    assert by_signature["heme_b+Cu_2+"] == "CU+HEM"
    assert by_signature["PQQ+Cu_2"] == "CU+PQQ"


def test_build_manifest_maps_targets_to_userccd_component_ids(tmp_path: Path) -> None:
    metadata_csv, list_path, cif_dir = _write_sample_set(tmp_path)
    userccd_manifest = tmp_path / "studio179_userccd_manifest.tsv"
    userccd_path = tmp_path / "studio179_all_components_userccd.cif"
    _write_userccd_manifest(userccd_manifest)
    userccd_path.write_text("data_S179002\n")

    manifest = build_studio179_manifest(
        sample_id_list=list_path,
        metadata_csv=metadata_csv,
        cif_dir=cif_dir,
        userccd_manifest_tsv=userccd_manifest,
        userccd_path=userccd_path,
    )
    by_target = dict(zip(manifest["disco_target"], manifest["af3_ligand_ccd_codes"]))

    assert by_target["Cu"] == "S179002"
    assert by_target["heme_b+Cu_2+"] == "S179079;S179002"
    assert by_target["PQQ+Cu_2"] == "S179010;S179002"
    assert by_target["FMN+Fe2S2"] == "S179052;S179060"
    assert by_target["PFOA+PFOS"] == "S179103;S179104"
    assert set(manifest["af3_user_ccd_path"]) == {str(userccd_path)}


def test_filter_manifest_by_target_excludes_multi_ligand_targets(tmp_path: Path) -> None:
    metadata_csv, list_path, cif_dir = _write_sample_set(tmp_path)
    manifest = build_studio179_manifest(
        sample_id_list=list_path,
        metadata_csv=metadata_csv,
        cif_dir=cif_dir,
    )

    cu_only = filter_studio179_manifest(manifest, targets="CU")
    hem_cu = filter_studio179_manifest(manifest, targets="HEM+CU")
    cu_hem = filter_studio179_manifest(manifest, targets="CU+HEM")

    assert cu_only["sample_id"].tolist() == [
        "length_150_Cu_sample_0",
        "length_150_Cu_sample_1",
    ]
    assert hem_cu["sample_id"].tolist() == ["length_150_heme_b+Cu_2+_sample_0"]
    assert cu_hem["sample_id"].tolist() == ["length_150_heme_b+Cu_2+_sample_0"]
    assert normalize_target_signatures(manifest, "HEM+CU,PQQ+CU") == ["CU+HEM", "CU+PQQ"]


def test_filter_manifest_by_length_and_sample_index(tmp_path: Path) -> None:
    metadata_csv, list_path, cif_dir = _write_sample_set(tmp_path)
    manifest = build_studio179_manifest(
        sample_id_list=list_path,
        metadata_csv=metadata_csv,
        cif_dir=cif_dir,
    )

    selected = filter_studio179_manifest(
        manifest,
        targets="HEM+TPP",
        lengths=[150, 250],
        sample_indices=[0, 3],
    )

    assert selected["sample_id"].tolist() == [
        "length_150_heme_b+TPP_sample_0",
        "length_250_heme_b+TPP_sample_0",
        "length_250_heme_b+TPP_sample_3",
    ]


def test_build_manifest_fails_for_missing_cif(tmp_path: Path) -> None:
    metadata_csv, list_path, cif_dir = _write_sample_set(tmp_path)
    (cif_dir / "length_150_cisplatin_sample_3.cif").unlink()

    with pytest.raises(FileNotFoundError, match="Missing 1 CIF files"):
        build_studio179_manifest(
            sample_id_list=list_path,
            metadata_csv=metadata_csv,
            cif_dir=cif_dir,
        )


def test_prepare_lc_seq_des_inputs_writes_name_list_manifest_and_summary(tmp_path: Path) -> None:
    metadata_csv, list_path, cif_dir = _write_sample_set(tmp_path)
    userccd_manifest = tmp_path / "studio179_userccd_manifest.tsv"
    userccd_path = tmp_path / "studio179_all_components_userccd.cif"
    out_dir = tmp_path / "prepared"
    _write_userccd_manifest(userccd_manifest)
    userccd_path.write_text("data_S179002\n")

    args = parse_args(
        [
            "--sample-id-list", str(list_path),
            "--metadata-csv", str(metadata_csv),
            "--cif-dir", str(cif_dir),
            "--user-ccd-manifest", str(userccd_manifest),
            "--user-ccd-path", str(userccd_path),
            "--out-dir", str(out_dir),
            "--target", "CU",
            "--length", "150",
            "--sample-index", "0,1",
            "--run-id", "smoke",
        ]
    )
    summary = prepare_lc_seq_des_inputs(args)

    sample_id_list = out_dir / "studio179_smoke.txt"
    manifest_tsv = out_dir / "studio179_smoke_manifest.tsv"
    summary_json = out_dir / "studio179_smoke_summary.json"

    assert summary["selected_sample_count"] == 2
    assert summary["selected_af3_component_ids"] == ["S179002"]
    assert sample_id_list.read_text().splitlines() == [
        "length_150_Cu_sample_0",
        "length_150_Cu_sample_1",
    ]
    manifest = pd.read_csv(manifest_tsv, sep="\t", keep_default_na=False)
    assert manifest["af3_ligand_ccd_codes"].tolist() == ["S179002", "S179002"]
    assert summary_json.exists()


def test_lc_seq_des_multi_studio179_annotation_config_updates_sample_dict(tmp_path: Path) -> None:
    manifest_tsv = tmp_path / "selected_manifest.tsv"
    userccd_path = tmp_path / "studio179_all_components_userccd.cif"
    pd.DataFrame(
        [
            {
                "sample_id": "length_150_Cu_sample_0",
                "af3_ligand_ccd_codes": "S179002",
            }
        ]
    ).to_csv(manifest_tsv, sep="\t", index=False)
    userccd_path.write_text("data_S179002\n")

    cfg = OmegaConf.create(
        {
            "studio179_userccd": {
                "enabled": True,
                "selected_manifest_tsv": str(manifest_tsv),
                "user_ccd_path": str(userccd_path),
            }
        }
    )
    annotation = _load_studio179_userccd_annotation(cfg)
    sample_dict = {
        "length_150_Cu_sample_0": {
            "pdb_chain_info": {
                "ligand_pn_unit_iids": ["B_1"],
            }
        }
    }

    annotated = _apply_studio179_userccd_annotation(sample_dict, annotation)

    assert annotated["length_150_Cu_sample_0"]["pdb_chain_info"]["af3_ligand_ccd_codes"] == ["S179002"]
    assert annotated["length_150_Cu_sample_0"]["pdb_chain_info"]["af3_user_ccd_path"] == str(userccd_path)


def test_resolve_studio179_paths_uses_studio179_root_manifest_layout(tmp_path: Path) -> None:
    class Args:
        disco_root = str(tmp_path / "DISCO_benchmark_data")
        studio179_root = None
        converted_root = None
        cif_dir = None
        sample_id_list = None
        metadata_csv = None
        userccd_manifest_tsv = None
        userccd_path = None

    paths = resolve_studio179_paths(Args())

    assert paths["studio179_root"] == (
        tmp_path
        / "DISCO_benchmark_data"
        / "disco_inference_benchmarks_release_data"
        / "studio-179"
    )
    assert paths["converted_root"] == paths["studio179_root"]
    assert paths["cif_dir"] == paths["studio179_root"] / "cifs"
    assert paths["sample_id_list"] == paths["converted_root"] / "studio179_all.txt"
    assert paths["metadata_csv"] == paths["studio179_root"] / "all_diversity_results.csv"
    assert paths["userccd_manifest_tsv"] == paths["converted_root"] / "conformer_cifs/studio179_userccd_manifest.tsv"
    assert paths["userccd_path"] == paths["converted_root"] / "conformer_cifs/studio179_all_components_userccd.cif"


def test_parse_args_loads_yaml_config_and_allows_cli_override(tmp_path: Path) -> None:
    config_path = tmp_path / "studio179.yaml"
    config_path.write_text(
        "\n".join(
            [
                "target: CU,MN",
                "length: '150,250'",
                "sample_index: '0,1'",
                "step: 123",
                "af3: true",
                "include_bonds: false",
                "user_ccd_path: /tmp/studio179.cif",
                "pocket_distance_for_docking_metrics: 8.0",
                "model_name: ignored_by_prepare",
            ]
        )
        + "\n"
    )

    args = parse_args(["--config", str(config_path), "--target", "HEM+CU"])

    assert args.config == str(config_path)
    assert args.target == "HEM+CU"
    assert args.length == "150,250"
    assert args.sample_index == "0,1"
    assert args.userccd_path == "/tmp/studio179.cif"
    assert args.ignored_config_keys == [
        "af3",
        "include_bonds",
        "model_name",
        "pocket_distance_for_docking_metrics",
        "step",
    ]


def test_make_artifact_stem_isolates_array_task_outputs() -> None:
    assert make_artifact_stem(run_id="target_HEM", array_id=None, num_arrays=None) == "studio179_target_HEM"
    assert (
        make_artifact_stem(run_id="target_HEM", array_id=3, num_arrays=8)
        == "studio179_target_HEM_array_3_of_8"
    )
