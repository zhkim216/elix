from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd
from alphafold3.cpp import cif_dict
from alphafold3.data.tools import rdkit_utils
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


AF3_USER_CCD_REQUIRED_KEYS = frozenset(
    {
        "_chem_comp.id",
        "_chem_comp.name",
        "_chem_comp.type",
        "_chem_comp.formula",
        "_chem_comp.mon_nstd_parent_comp_id",
        "_chem_comp.pdbx_synonyms",
        "_chem_comp.formula_weight",
        "_chem_comp_atom.comp_id",
        "_chem_comp_atom.atom_id",
        "_chem_comp_atom.type_symbol",
        "_chem_comp_atom.charge",
        "_chem_comp_atom.pdbx_model_Cartn_x_ideal",
        "_chem_comp_atom.pdbx_model_Cartn_y_ideal",
        "_chem_comp_atom.pdbx_model_Cartn_z_ideal",
        "_chem_comp_bond.atom_id_1",
        "_chem_comp_bond.atom_id_2",
        "_chem_comp_bond.value_order",
        "_chem_comp_bond.pdbx_aromatic_flag",
    }
)


@dataclass(frozen=True)
class ConversionJob:
    component_id: str
    ligand: str
    ligand_name: str
    metadata_ccd: str
    priority: str
    sdf_path: Path
    cif_path: Path


@dataclass
class ConversionRecord:
    component_id: str
    ligand: str
    ligand_name: str
    metadata_ccd: str
    priority: str
    sdf_path: str
    cif_path: str
    smiles: str
    num_atoms: int
    num_non_h_atoms: int
    num_bonds: int
    formal_charge: int
    stereoany_bonds_normalized: int
    status: str
    error: str


def sanitize_filename_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.()+-]+", "_", value.strip())
    stem = stem.strip("._")
    return stem or "ligand"


def make_component_id(prefix: str, index: int) -> str:
    component_id = f"{prefix}{index:03d}"
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9-]*", component_id):
        raise ValueError(
            "Component IDs must start with a letter and contain only letters, "
            "numbers, or hyphens. Avoid underscores for AF3 userCCD entries."
        )
    return component_id


def _read_metadata(metadata_csv: Path) -> pd.DataFrame:
    metadata_df = pd.read_csv(metadata_csv, keep_default_na=False)
    required_columns = {"ligand", "ligand_name", "CCD"}
    missing_columns = required_columns - set(metadata_df.columns)
    if missing_columns:
        raise ValueError(f"Metadata CSV is missing columns: {sorted(missing_columns)}")
    return metadata_df


def _find_sdfs(sdf_root: Path) -> dict[str, Path]:
    sdf_paths = sorted(sdf_root.glob("priority_*/*.sdf"))
    if not sdf_paths:
        raise FileNotFoundError(f"No Studio-179 SDF files found under {sdf_root}")

    by_stem: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in sdf_paths:
        if path.stem in by_stem:
            duplicates.append(path.stem)
        by_stem[path.stem] = path
    if duplicates:
        raise ValueError(f"Duplicate SDF stems found: {sorted(set(duplicates))}")
    return by_stem


def _priority_for_sdf(path: Path) -> str:
    match = re.fullmatch(r"priority_(.+)", path.parent.name)
    return match.group(1) if match else path.parent.name


def _deduplicated_cif_path(output_dir: Path, stem: str, used_stems: set[str]) -> Path:
    safe_stem = sanitize_filename_stem(stem)
    candidate = safe_stem
    index = 2
    while candidate in used_stems:
        candidate = f"{safe_stem}_{index}"
        index += 1
    used_stems.add(candidate)
    return output_dir / f"{candidate}.cif"


def build_conversion_jobs(
    *,
    sdf_root: Path,
    metadata_csv: Path,
    output_dir: Path,
    component_prefix: str = "S179",
) -> list[ConversionJob]:
    metadata_df = _read_metadata(metadata_csv)
    sdf_by_stem = _find_sdfs(sdf_root)

    jobs: list[ConversionJob] = []
    used_sdf_stems: set[str] = set()
    used_output_stems: set[str] = set()

    for _, row in metadata_df.iterrows():
        ligand = str(row["ligand"]).strip()
        if "+" in ligand or ligand not in sdf_by_stem:
            continue

        sdf_path = sdf_by_stem[ligand]
        used_sdf_stems.add(ligand)
        ligand_name = str(row["ligand_name"]).strip() or ligand
        metadata_ccd = str(row["CCD"]).strip()
        priority = str(row["priority"]).strip() if "priority" in row.index else ""
        if not priority:
            priority = _priority_for_sdf(sdf_path)
        jobs.append(
            ConversionJob(
                component_id=make_component_id(component_prefix, len(jobs) + 1),
                ligand=ligand,
                ligand_name=ligand_name,
                metadata_ccd=metadata_ccd,
                priority=priority,
                sdf_path=sdf_path,
                cif_path=_deduplicated_cif_path(
                    output_dir, ligand_name, used_output_stems
                ),
            )
        )

    for sdf_stem, sdf_path in sorted(sdf_by_stem.items()):
        if sdf_stem in used_sdf_stems:
            continue
        jobs.append(
            ConversionJob(
                component_id=make_component_id(component_prefix, len(jobs) + 1),
                ligand=sdf_stem,
                ligand_name=sdf_stem,
                metadata_ccd="",
                priority=_priority_for_sdf(sdf_path),
                sdf_path=sdf_path,
                cif_path=_deduplicated_cif_path(output_dir, sdf_stem, used_output_stems),
            )
        )

    return jobs


def normalize_stereoany_bonds(mol: Chem.Mol) -> int:
    count = 0
    for bond in mol.GetBonds():
        if bond.GetStereo() == Chem.BondStereo.STEREOANY:
            bond.SetStereo(Chem.BondStereo.STEREONONE)
            bond.SetBondDir(Chem.BondDir.NONE)
            count += 1
    return count


def _mol_formula(mol: Chem.Mol) -> str:
    try:
        return rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        return "?"


def _mol_formula_weight(mol: Chem.Mol) -> str:
    try:
        return f"{Descriptors.MolWt(mol):.3f}"
    except Exception:
        return "?"


def _canonical_smiles(mol: Chem.Mol) -> str:
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return ""


def _augment_for_af3_user_ccd(
    *,
    mol_cif: cif_dict.CifDict,
    mol: Chem.Mol,
    component_id: str,
    ligand_name: str,
) -> cif_dict.CifDict:
    data = mol_cif.to_dict()
    num_atoms = len(data.get("_chem_comp_atom.atom_id", []))
    if num_atoms == 0:
        raise ValueError(f"{component_id} has no atoms after hydrogen filtering")

    for coord_key in (
        "_chem_comp_atom.pdbx_model_Cartn_x_ideal",
        "_chem_comp_atom.pdbx_model_Cartn_y_ideal",
        "_chem_comp_atom.pdbx_model_Cartn_z_ideal",
    ):
        if len(data.get(coord_key, [])) != num_atoms:
            raise ValueError(
                f"{component_id} is missing complete ideal coordinates in {coord_key}"
            )

    data["_chem_comp.id"] = [component_id]
    data["_chem_comp.name"] = [ligand_name or component_id]
    data["_chem_comp.type"] = ["non-polymer"]
    data["_chem_comp.formula"] = [_mol_formula(mol)]
    data["_chem_comp.mon_nstd_parent_comp_id"] = ["?"]
    data["_chem_comp.pdbx_synonyms"] = ["?"]
    data["_chem_comp.formula_weight"] = [_mol_formula_weight(mol)]
    data.setdefault("_chem_comp_atom.pdbx_leaving_atom_flag", ["N"] * num_atoms)
    data.setdefault("_chem_comp_bond.atom_id_1", [])
    data.setdefault("_chem_comp_bond.atom_id_2", [])
    data.setdefault("_chem_comp_bond.value_order", [])
    data.setdefault("_chem_comp_bond.pdbx_aromatic_flag", [])

    return cif_dict.CifDict(data)


def sdf_to_user_ccd_text(
    *,
    sdf_path: Path,
    component_id: str,
    ligand_name: str,
    include_hydrogens: bool = False,
) -> tuple[str, ConversionRecord]:
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = supplier[0] if supplier else None
    if mol is None:
        raise ValueError(f"RDKit could not read SDF: {sdf_path}")
    if mol.GetNumConformers() == 0:
        raise ValueError(f"SDF does not contain conformer coordinates: {sdf_path}")

    mol = Chem.Mol(mol)
    stereoany_count = normalize_stereoany_bonds(mol)
    smiles = _canonical_smiles(mol)
    mol_cif = rdkit_utils.mol_to_ccd_cif(
        mol,
        component_id=component_id,
        pdbx_smiles=smiles or None,
        include_hydrogens=include_hydrogens,
    )
    user_ccd_cif = _augment_for_af3_user_ccd(
        mol_cif=mol_cif,
        mol=mol,
        component_id=component_id,
        ligand_name=ligand_name,
    )
    text = user_ccd_cif.to_string()
    validate_user_ccd_text(text, component_id=component_id)

    record = ConversionRecord(
        component_id=component_id,
        ligand=sdf_path.stem,
        ligand_name=ligand_name,
        metadata_ccd="",
        priority=_priority_for_sdf(sdf_path),
        sdf_path=str(sdf_path),
        cif_path="",
        smiles=smiles,
        num_atoms=int(mol.GetNumAtoms()),
        num_non_h_atoms=sum(
            1 for atom in mol.GetAtoms() if atom.GetSymbol() not in {"H", "D"}
        ),
        num_bonds=int(mol.GetNumBonds()),
        formal_charge=sum(atom.GetFormalCharge() for atom in mol.GetAtoms()),
        stereoany_bonds_normalized=stereoany_count,
        status="ok",
        error="",
    )
    return text, record


def validate_user_ccd_text(cif_text: str, *, component_id: str) -> None:
    parsed = cif_dict.parse_multi_data_cif(cif_text)
    if component_id not in parsed:
        raise ValueError(f"userCCD text does not contain component {component_id}")
    component_cif = parsed[component_id]
    missing_keys = AF3_USER_CCD_REQUIRED_KEYS - set(component_cif.keys())
    if missing_keys:
        raise ValueError(
            f"Component {component_id} is missing AF3 userCCD keys: {sorted(missing_keys)}"
        )


def _write_text(path: Path, text: str, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing file without --force: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(text)
    tmp_path.replace(path)


def _record_for_error(job: ConversionJob, error: Exception) -> ConversionRecord:
    return ConversionRecord(
        component_id=job.component_id,
        ligand=job.ligand,
        ligand_name=job.ligand_name,
        metadata_ccd=job.metadata_ccd,
        priority=job.priority,
        sdf_path=str(job.sdf_path),
        cif_path=str(job.cif_path),
        smiles="",
        num_atoms=0,
        num_non_h_atoms=0,
        num_bonds=0,
        formal_charge=0,
        stereoany_bonds_normalized=0,
        status="error",
        error=f"{type(error).__name__}: {error}",
    )


def convert_studio179_sdfs(
    *,
    sdf_root: Path,
    metadata_csv: Path,
    output_dir: Path,
    manifest_path: Path,
    combined_cif_path: Path,
    component_prefix: str = "S179",
    include_hydrogens: bool = False,
    force: bool = False,
    dry_run: bool = False,
    allow_errors: bool = False,
) -> dict[str, object]:
    jobs = build_conversion_jobs(
        sdf_root=sdf_root,
        metadata_csv=metadata_csv,
        output_dir=output_dir,
        component_prefix=component_prefix,
    )
    if not jobs:
        raise ValueError("No conversion jobs were built")

    records: list[ConversionRecord] = []
    combined_blocks: list[str] = []
    for job in jobs:
        try:
            cif_text, record = sdf_to_user_ccd_text(
                sdf_path=job.sdf_path,
                component_id=job.component_id,
                ligand_name=job.ligand_name,
                include_hydrogens=include_hydrogens,
            )
            record.ligand = job.ligand
            record.metadata_ccd = job.metadata_ccd
            record.priority = job.priority
            record.cif_path = str(job.cif_path)
            records.append(record)
            combined_blocks.append(cif_text.rstrip() + "\n")
            if not dry_run:
                _write_text(job.cif_path, cif_text, force=force)
        except Exception as exc:
            records.append(_record_for_error(job, exc))

    record_dicts = [asdict(record) for record in records]
    error_count = sum(1 for record in records if record.status != "ok")
    if not dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(record_dicts).to_csv(manifest_path, sep="\t", index=False)
        if combined_blocks:
            validate_combined_user_ccd_text(
                "".join(combined_blocks), expected_count=len(combined_blocks)
            )
            _write_text(combined_cif_path, "".join(combined_blocks), force=force)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "sdf_root": str(sdf_root),
        "metadata_csv": str(metadata_csv),
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "combined_cif_path": str(combined_cif_path),
        "component_prefix": component_prefix,
        "include_hydrogens": include_hydrogens,
        "job_count": len(jobs),
        "converted_count": len(jobs) - error_count,
        "error_count": error_count,
        "stereoany_bonds_normalized_total": sum(
            record.stereoany_bonds_normalized for record in records
        ),
        "stereoany_ligands": [
            record.ligand for record in records if record.stereoany_bonds_normalized
        ],
    }
    if error_count and not allow_errors:
        error_preview = "\n".join(
            f"{record.ligand}: {record.error}"
            for record in records
            if record.status != "ok"
        )
        raise RuntimeError(f"{error_count} SDF conversion(s) failed:\n{error_preview}")
    return summary


def validate_combined_user_ccd_text(
    cif_text: str, *, expected_count: int | None = None
) -> None:
    parsed = cif_dict.parse_multi_data_cif(cif_text)
    if expected_count is not None and len(parsed) != expected_count:
        raise ValueError(
            f"Combined userCCD contains {len(parsed)} components, expected {expected_count}"
        )
    for component_id, component_cif in parsed.items():
        missing_keys = AF3_USER_CCD_REQUIRED_KEYS - set(component_cif.keys())
        if missing_keys:
            raise ValueError(
                f"Component {component_id} is missing AF3 userCCD keys: {sorted(missing_keys)}"
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Studio-179 SDF conformers to AF3 userCCD mmCIF files."
    )
    studio179_data_root = (
        Path.home()
        / "model-dev/datasets/val_cifs/DISCO_benchmark_data/"
        / "disco_inference_benchmarks_release_data/studio-179"
    )
    parser.add_argument(
        "--sdf-root", default=str(Path.home() / "model-dev/DISCO/studio-179")
    )
    parser.add_argument(
        "--metadata-csv",
        default=str(studio179_data_root / "all_diversity_results.csv"),
    )
    parser.add_argument(
        "--output-dir", default=str(studio179_data_root / "conformer_cifs")
    )
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--combined-cif-path", default=None)
    parser.add_argument("--component-prefix", default="S179")
    parser.add_argument(
        "--include-hydrogens", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing CIF outputs."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Convert and validate without writing files.",
    )
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="Write successful outputs even if some SDFs fail.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_dir = Path(args.output_dir).expanduser()
    manifest_path = (
        Path(args.manifest_path).expanduser()
        if args.manifest_path
        else output_dir / "studio179_userccd_manifest.tsv"
    )
    combined_cif_path = (
        Path(args.combined_cif_path).expanduser()
        if args.combined_cif_path
        else output_dir / "studio179_all_components_userccd.cif"
    )
    summary = convert_studio179_sdfs(
        sdf_root=Path(args.sdf_root).expanduser(),
        metadata_csv=Path(args.metadata_csv).expanduser(),
        output_dir=output_dir,
        manifest_path=manifest_path,
        combined_cif_path=combined_cif_path,
        component_prefix=args.component_prefix,
        include_hydrogens=args.include_hydrogens,
        force=args.force,
        dry_run=args.dry_run,
        allow_errors=args.allow_errors,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
