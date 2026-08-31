from __future__ import annotations

import argparse
import json
import os
import shutil
import string
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.benchmarks.denovoval.sequence_design_inputs import (
    load_canonical_source,
    stage_row,
)
from allatom_design.eval.benchmarks.sequence_design.structure_bridge import (
    CcdAlias,
    allocate_ccd_aliases,
    sha256_file,
    write_ccd_aliases,
)


DEFAULT_CONFIG = Path(
    "/home/users/zhkim216/code/elix/allatom_design/configs/benchmarks/"
    "denovoval/ADFLIP/default.yaml"
)
MODES = ("per_ccd_smoke", "full")
BASE36 = string.digits + string.ascii_uppercase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage canonical denovoval CIFs as ADFLIP PDB inputs."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=MODES, default="per_ccd_smoke")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def cfg_path(cfg: DictConfig, key: str) -> Path:
    value = OmegaConf.select(cfg, key)
    if value is None:
        raise ValueError(f"Missing required config path: {key}")
    return Path(str(value)).expanduser()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    frame.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def select_mode_rows(source: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "full":
        return source.copy()
    if mode != "per_ccd_smoke":
        raise ValueError(f"Unsupported mode: {mode}")
    selected = (
        source.sort_values(["ccd_code", "protein_length", "replicate_index"])
        .groupby("ccd_code", sort=True, as_index=False)
        .head(1)
        .sort_values("source_index")
        .reset_index(drop=True)
    )
    invalid = selected.loc[
        (selected["protein_length"] != 150) | (selected["replicate_index"] != 0),
        "sample_id",
    ].tolist()
    if invalid:
        raise ValueError(
            "Per-CCD representatives must be deterministic *_len150_0 samples: "
            f"{invalid[:20]}"
        )
    return selected


def allocate_adflip_aliases(
    ccd_codes: pd.Series,
    *,
    force_alias_codes: list[str],
) -> list[CcdAlias]:
    """Add aliases for CCDs that ADFLIP's PDB parser would otherwise discard."""
    normalized = {str(code).strip().upper() for code in ccd_codes}
    requested = sorted({str(code).strip().upper() for code in force_alias_codes})
    missing = sorted(set(requested) - normalized)
    if missing:
        raise ValueError(f"Forced ADFLIP CCD aliases are absent from the dataset: {missing}")

    aliases = allocate_ccd_aliases(normalized)
    aliased_codes = {alias.original_ccd_code for alias in aliases}
    reserved = normalized | {alias.pdb_ccd_code for alias in aliases}
    candidates = (
        f"Z{first}{second}"
        for first in BASE36
        for second in BASE36
        if f"Z{first}{second}" not in reserved
    )
    for code in requested:
        if code in aliased_codes:
            continue
        alias = next(candidates)
        reserved.add(alias)
        aliases.append(CcdAlias(code, alias, "adflip_parser_exclusion"))
    return sorted(aliases, key=lambda alias: alias.original_ccd_code)


def require_per_ccd_gate(cfg: DictConfig) -> None:
    report_path = cfg_path(cfg, "backmapping.per_ccd_smoke.validation_json")
    if not report_path.is_file():
        raise FileNotFoundError(
            "Full staging requires a complete per-CCD backmapping gate: "
            f"{report_path}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise ValueError(
            "Full staging requires per-CCD backmapping status=complete: "
            f"{report_path}"
        )
    diagnostics = report.get("diagnostics", {})
    nonzero = {key: value for key, value in diagnostics.items() if value != 0}
    if nonzero:
        raise ValueError(
            "Full staging requires zero per-CCD backmapping diagnostics: "
            f"{nonzero}"
        )
    expected_rows = int(cfg.dataset.expected_unique_ccd_count) * int(
        cfg.sequence_design.expected_designs_per_sample
    )
    if int(report.get("observed_rows", -1)) != expected_rows:
        raise ValueError("Per-CCD backmapping row count does not match the config")
    manifest_path = Path(str(report.get("design_manifest", "")))
    manifest_digest = str(report.get("design_manifest_sha256", ""))
    if not manifest_path.is_file() or sha256_file(manifest_path) != manifest_digest:
        raise ValueError("Per-CCD backmapping manifest is missing or has drifted")


def copy_source_cif(source: Path, destination: Path, *, overwrite: bool) -> str:
    expected_digest = sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        observed_digest = sha256_file(destination)
        if observed_digest != expected_digest:
            raise ValueError(
                f"Existing staged CIF digest mismatch: {destination}: "
                f"{observed_digest} != {expected_digest}"
            )
        return "reused"
    tmp_path = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    shutil.copyfile(source, tmp_path)
    os.replace(tmp_path, destination)
    if sha256_file(destination) != expected_digest:
        raise ValueError(f"Staged CIF copy digest mismatch: {destination}")
    return "copied"


def mode_paths(cfg: DictConfig, mode: str) -> dict[str, Path]:
    prefix = f"staging.{mode}"
    return {
        "root": cfg_path(cfg, f"{prefix}.root"),
        "source_cif_dir": cfg_path(cfg, f"{prefix}.source_cif_dir"),
        "pdb_dir": cfg_path(cfg, f"{prefix}.pdb_dir"),
        "manifest_csv": cfg_path(cfg, f"{prefix}.manifest_csv"),
        "validation_json": cfg_path(cfg, f"{prefix}.validation_json"),
    }


def prepare_mode(cfg: DictConfig, *, mode: str, overwrite: bool) -> dict[str, Any]:
    source, source_diagnostics = load_canonical_source(cfg)
    selected = select_mode_rows(source, mode)
    expected_count = int(
        cfg.dataset.expected_count
        if mode == "full"
        else cfg.dataset.expected_unique_ccd_count
    )
    if len(selected) != expected_count:
        raise ValueError(
            f"Selected {len(selected)} rows for {mode}, expected {expected_count}"
        )
    if mode == "full":
        require_per_ccd_gate(cfg)

    paths = mode_paths(cfg, mode)
    for key in ("root", "source_cif_dir", "pdb_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)

    aliases = allocate_adflip_aliases(
        source["ccd_code"],
        force_alias_codes=list(cfg.adflip.force_ligand_aliases),
    )
    ccd_mapping_csv = cfg_path(cfg, "staging.ccd_mapping_csv")
    write_ccd_aliases(ccd_mapping_csv, aliases)
    alias_by_original = {
        alias.original_ccd_code: alias.pdb_ccd_code for alias in aliases
    }

    selected_ids = set(map(str, selected["sample_id"]))
    stale_cifs = sorted(
        str(path)
        for path in paths["source_cif_dir"].glob("*.cif")
        if path.stem not in selected_ids
    )
    stale_pdbs = sorted(
        str(path)
        for path in paths["pdb_dir"].glob("*.pdb")
        if path.stem not in selected_ids
    )
    if stale_cifs or stale_pdbs:
        raise ValueError(
            "Staging contains stale files outside the selected source contract: "
            f"cifs={stale_cifs[:10]}, pdbs={stale_pdbs[:10]}"
        )

    rows: list[dict[str, Any]] = []
    for _, canonical_row in selected.iterrows():
        sample_id = str(canonical_row["sample_id"])
        canonical_path = Path(str(canonical_row["source_cif_path"]))
        staged_cif_path = paths["source_cif_dir"] / f"{sample_id}.cif"
        try:
            copy_status = copy_source_cif(
                canonical_path,
                staged_cif_path,
                overwrite=overwrite,
            )
            staged_row = canonical_row.copy()
            staged_row["source_cif_path"] = str(staged_cif_path)
            result = stage_row(
                staged_row,
                staged_pdb_dir=paths["pdb_dir"],
                alias_by_original=alias_by_original,
                protein_chain_id=str(cfg.structure.protein_chain_id),
                ligand_chain_id=str(cfg.structure.ligand_chain_id),
                expected_ligand_residue_id=(
                    int(cfg.structure.ligand_residue_id)
                    if OmegaConf.select(cfg, "structure.ligand_residue_id")
                    is not None
                    else None
                ),
                overwrite=overwrite,
            )
            result.update(
                {
                    "canonical_source_cif_path": str(canonical_path),
                    "canonical_source_cif_sha256": sha256_file(canonical_path),
                    "staged_source_cif_status": copy_status,
                }
            )
        except Exception as exc:
            result = {
                "source_index": int(canonical_row["source_index"]),
                "sample_id": sample_id,
                "kind": str(canonical_row["kind"]),
                "protein_length": int(canonical_row["protein_length"]),
                "ccd_code": str(canonical_row["ccd_code"]),
                "pdb_ccd_code": alias_by_original.get(
                    str(canonical_row["ccd_code"]),
                    str(canonical_row["ccd_code"]),
                ),
                "protein_chain_id": str(cfg.structure.protein_chain_id),
                "ligand_chain_id": str(cfg.structure.ligand_chain_id),
                "canonical_source_cif_path": str(canonical_path),
                "source_cif_path": str(staged_cif_path),
                "staged_pdb_path": str(paths["pdb_dir"] / f"{sample_id}.pdb"),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(result)

    frame = pd.DataFrame(rows).sort_values("source_index").reset_index(drop=True)
    atomic_write_csv(paths["manifest_csv"], frame)
    failed = frame[frame["status"] == "failed"]
    report: dict[str, Any] = {
        "status": "complete" if failed.empty and len(frame) == expected_count else "failed",
        "mode": mode,
        **source_diagnostics,
        "selected_row_count": len(frame),
        "selected_unique_ccd_count": int(frame["ccd_code"].nunique()),
        "manifest_csv": str(paths["manifest_csv"]),
        "manifest_sha256": sha256_file(paths["manifest_csv"]),
        "ccd_mapping_csv": str(ccd_mapping_csv),
        "ccd_mapping_sha256": sha256_file(ccd_mapping_csv),
        "alias_count": len(aliases),
        "status_counts": dict(Counter(frame["status"])),
        "diagnostics": {
            "missing": int(frame["error"].astype(str).str.contains("FileNotFoundError").sum()),
            "failed": len(failed),
            "skipped": 0,
            "unmapped": 0,
            "stale_cif": len(stale_cifs),
            "stale_pdb": len(stale_pdbs),
            "categorical_mismatches": int(
                frame.get("categorical_mismatch_count", pd.Series(dtype=int))
                .fillna(0)
                .sum()
            ),
            "auth_chain_mismatches": int(
                frame.get("auth_chain_mismatch_count", pd.Series(dtype=int))
                .fillna(0)
                .sum()
            ),
            "auth_residue_mismatches": int(
                frame.get("auth_residue_mismatch_count", pd.Series(dtype=int))
                .fillna(0)
                .sum()
            ),
        },
        "failed_examples": failed[["sample_id", "error"]]
        .head(20)
        .to_dict(orient="records"),
        "contract": {
            "canonical_membership": str(cfg.dataset.sample_id_list),
            "staged_source_is_authoritative_after_copy": True,
            "loader": "atomworks.read_any -> get_structure",
            "protein_chain_id": str(cfg.structure.protein_chain_id),
            "ligand_chain_id": str(cfg.structure.ligand_chain_id),
            "per_ccd_selection": "one deterministic *_len150_0 per CCD",
        },
    }
    atomic_write_json(paths["validation_json"], report)
    if report["status"] != "complete":
        raise RuntimeError(f"Staging failed; see {paths['validation_json']}")
    return report


def main() -> None:
    args = parse_args()
    report = prepare_mode(
        OmegaConf.load(args.config),
        mode=args.mode,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
