from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


SAMPLE_ID_PATTERN = re.compile(
    r"^length_(?P<length>\d+)_(?P<disco_target>.+)_sample_(?P<sample_index>\d+)$"
)

COMPONENT_ALIASES = {
    "acyl-coa": "acetyl_coenzyme_a",
    "cu2": "cu",
    "cu_2": "cu",
    "cu_2+": "cu",
    "fe2s2": "fe2-s2",
    "fmn": "flavin_mononucleotide",
    "pqq": "pyrroloquinoline_quinone",
    "sam": "s-adenosylmethionine",
    "tpp": "thiamine_diphosphate",
}


@dataclass(frozen=True)
class Studio179SampleId:
    sample_id: str
    length: int
    disco_target: str
    sample_index: int


@dataclass(frozen=True)
class TargetCcdMapping:
    disco_target: str
    ccd_codes: tuple[str, ...]
    metadata_ligands: tuple[str, ...]
    metadata_ligand_names: tuple[str, ...]
    mapping_source: str


def parse_studio179_sample_id(sample_id: str) -> Studio179SampleId:
    match = SAMPLE_ID_PATTERN.fullmatch(sample_id)
    if match is None:
        raise ValueError(f"Invalid Studio-179 sample_id: {sample_id}")
    return Studio179SampleId(
        sample_id=sample_id,
        length=int(match.group("length")),
        disco_target=match.group("disco_target"),
        sample_index=int(match.group("sample_index")),
    )


def read_sample_ids(path: str | Path) -> list[str]:
    sample_ids = [line.strip() for line in Path(path).read_text().splitlines()]
    return [sample_id for sample_id in sample_ids if sample_id]


def normalize_disco_key(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    for suffix in ("_final_0", "_ideal", "_definition"):
        text = text.replace(suffix, "")
    return text.lower()


def normalize_ccd_codes(value: Iterable[str] | str | None) -> list[str]:
    if value is None:
        return []
    raw_values = [value] if isinstance(value, str) else list(value)

    codes: list[str] = []
    for raw_value in raw_values:
        for part in re.split(r"[,;|\s]+", str(raw_value)):
            code = part.strip().upper()
            if not code or code in {"NAN", "NONE", "NULL"}:
                continue
            if code not in codes:
                codes.append(code)
    return codes


def format_ccd_codes(codes: Iterable[str]) -> str:
    return ";".join(normalize_ccd_codes(codes))


def split_semicolon_values(value: object) -> list[str]:
    values: list[str] = []
    for part in str(value or "").split(";"):
        item = part.strip()
        if item and item not in values:
            values.append(item)
    return values


def _split_comma_values(value: Iterable[str] | str | None) -> list[str]:
    if value is None:
        return []
    raw_values = [value] if isinstance(value, str) else list(value)

    values: list[str] = []
    for raw_value in raw_values:
        for part in str(raw_value).split(","):
            item = part.strip()
            if item and item not in values:
                values.append(item)
    return values


def _fallback_target_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    return token.upper()


def _format_target_signature(tokens: Iterable[str]) -> str:
    normalized_tokens: list[str] = []
    for token in tokens:
        normalized_token = _fallback_target_token(token)
        if normalized_token and normalized_token not in normalized_tokens:
            normalized_tokens.append(normalized_token)
    return "+".join(sorted(normalized_tokens))


def _row_text(row: pd.Series, column: str) -> str:
    if column not in row.index or pd.isna(row[column]):
        return ""
    return str(row[column]).strip()


def _row_ccd_codes(row: pd.Series) -> list[str]:
    if "CCD" not in row.index or pd.isna(row["CCD"]):
        return []
    return normalize_ccd_codes(str(row["CCD"]))


def _build_metadata_lookup(metadata_df: pd.DataFrame) -> dict[str, pd.Series]:
    required_columns = {"ligand", "ligand_name", "CCD"}
    missing_columns = required_columns - set(metadata_df.columns)
    if missing_columns:
        raise ValueError(f"Metadata CSV is missing columns: {sorted(missing_columns)}")

    lookup: dict[str, pd.Series] = {}
    for _, row in metadata_df.iterrows():
        for column in ("ligand", "ligand_name"):
            key = normalize_disco_key(row[column])
            if key and key not in lookup:
                lookup[key] = row
    return lookup


def _build_userccd_lookup(userccd_manifest_tsv: str | Path) -> dict[str, str]:
    manifest_df = pd.read_csv(userccd_manifest_tsv, sep="\t", keep_default_na=False)
    required_columns = {"component_id", "ligand", "ligand_name", "status"}
    missing_columns = required_columns - set(manifest_df.columns)
    if missing_columns:
        raise ValueError(
            f"userCCD manifest is missing columns: {sorted(missing_columns)}"
        )

    failed_rows = manifest_df.loc[manifest_df["status"].astype(str) != "ok"]
    if not failed_rows.empty:
        raise ValueError(
            "userCCD manifest contains failed rows: "
            f"{failed_rows[['ligand', 'status', 'error']].head(5).to_dict('records')}"
        )

    lookup: dict[str, str] = {}
    for _, row in manifest_df.iterrows():
        component_id = str(row["component_id"]).strip()
        for column in ("ligand", "ligand_name", "metadata_ccd"):
            if column not in row.index:
                continue
            key = normalize_disco_key(row[column])
            if key and key not in lookup:
                lookup[key] = component_id
    return lookup


def _canonical_component_key(component: str) -> str:
    key = normalize_disco_key(component)
    return COMPONENT_ALIASES.get(key, key)


def _unique_extend(values: list[str], additions: Iterable[str]) -> None:
    for addition in additions:
        if addition and addition not in values:
            values.append(addition)


def _derive_component_mapping(
    target_or_metadata_ligand: str,
    lookup: dict[str, pd.Series],
) -> tuple[list[str], list[str], list[str], bool]:
    ccd_codes: list[str] = []
    metadata_ligands: list[str] = []
    metadata_ligand_names: list[str] = []
    all_components_resolved = True

    components = [part for part in normalize_disco_key(target_or_metadata_ligand).split("+") if part]
    for component in components:
        row = lookup.get(_canonical_component_key(component))
        if row is None:
            all_components_resolved = False
            continue
        _unique_extend(ccd_codes, _row_ccd_codes(row))
        _unique_extend(metadata_ligands, [_row_text(row, "ligand")])
        _unique_extend(metadata_ligand_names, [_row_text(row, "ligand_name")])

    return ccd_codes, metadata_ligands, metadata_ligand_names, all_components_resolved


def _target_signature_from_name(
    target_or_metadata_ligand: str,
    lookup: dict[str, pd.Series],
) -> str:
    tokens: list[str] = []
    components = [part for part in normalize_disco_key(target_or_metadata_ligand).split("+") if part]
    for component in components:
        row = lookup.get(_canonical_component_key(component))
        if row is None:
            tokens.append(component)
            continue

        ccd_codes = _row_ccd_codes(row)
        if ccd_codes:
            tokens.append(ccd_codes[0])
        else:
            tokens.append(_row_text(row, "ligand_name") or _row_text(row, "ligand") or component)

    return _format_target_signature(tokens)


def resolve_target_ccd_mapping(
    disco_target: str,
    metadata_lookup: dict[str, pd.Series],
) -> TargetCcdMapping:
    target_key = normalize_disco_key(disco_target)
    row = metadata_lookup.get(target_key)
    source_prefix = "metadata"

    if row is None:
        alias_key = COMPONENT_ALIASES.get(target_key)
        if alias_key:
            row = metadata_lookup.get(alias_key)
            source_prefix = "alias_metadata"

    if row is not None:
        ccd_codes = _row_ccd_codes(row)
        metadata_ligands = [_row_text(row, "ligand")]
        metadata_ligand_names = [_row_text(row, "ligand_name")]
        if ccd_codes:
            return TargetCcdMapping(
                disco_target=disco_target,
                ccd_codes=tuple(ccd_codes),
                metadata_ligands=tuple(metadata_ligands),
                metadata_ligand_names=tuple(metadata_ligand_names),
                mapping_source=f"{source_prefix}_ccd",
            )

        for source_value in (disco_target, _row_text(row, "ligand"), _row_text(row, "ligand_name")):
            if "+" not in str(source_value):
                continue
            component_codes, component_ligands, component_names, resolved = _derive_component_mapping(
                str(source_value),
                metadata_lookup,
            )
            if resolved:
                return TargetCcdMapping(
                    disco_target=disco_target,
                    ccd_codes=tuple(component_codes),
                    metadata_ligands=tuple(component_ligands or metadata_ligands),
                    metadata_ligand_names=tuple(component_names or metadata_ligand_names),
                    mapping_source=(
                        f"{source_prefix}_component_ccd"
                        if component_codes
                        else f"{source_prefix}_component_no_ccd"
                    ),
                )

        return TargetCcdMapping(
            disco_target=disco_target,
            ccd_codes=(),
            metadata_ligands=tuple(metadata_ligands),
            metadata_ligand_names=tuple(metadata_ligand_names),
            mapping_source=f"{source_prefix}_no_ccd",
        )

    if "+" in disco_target:
        ccd_codes, metadata_ligands, metadata_ligand_names, resolved = _derive_component_mapping(
            disco_target,
            metadata_lookup,
        )
        if resolved:
            return TargetCcdMapping(
                disco_target=disco_target,
                ccd_codes=tuple(ccd_codes),
                metadata_ligands=tuple(metadata_ligands),
                metadata_ligand_names=tuple(metadata_ligand_names),
                mapping_source="alias_component_ccd" if ccd_codes else "alias_component_no_ccd",
            )

    raise ValueError(f"Could not map Studio-179 DISCO target to metadata: {disco_target}")


def resolve_target_userccd_mapping(
    ccd_mapping: TargetCcdMapping,
    userccd_lookup: dict[str, str],
) -> tuple[tuple[str, ...], str]:
    component_ids: list[str] = []
    missing: list[str] = []
    max_items = max(
        len(ccd_mapping.metadata_ligands),
        len(ccd_mapping.metadata_ligand_names),
    )
    for index in range(max_items):
        metadata_ligand = (
            ccd_mapping.metadata_ligands[index]
            if index < len(ccd_mapping.metadata_ligands)
            else ""
        )
        metadata_ligand_name = (
            ccd_mapping.metadata_ligand_names[index]
            if index < len(ccd_mapping.metadata_ligand_names)
            else ""
        )
        component_id = (
            userccd_lookup.get(normalize_disco_key(metadata_ligand))
            or userccd_lookup.get(normalize_disco_key(metadata_ligand_name))
        )
        if component_id is None:
            missing.append(metadata_ligand or metadata_ligand_name)
            continue
        component_ids.append(component_id)

    if missing:
        raise ValueError(
            "Could not map Studio-179 target to userCCD component IDs: "
            f"{ccd_mapping.disco_target}; missing={missing}"
        )

    return tuple(component_ids), f"{ccd_mapping.mapping_source}_userccd"


def build_studio179_manifest(
    *,
    sample_id_list: str | Path,
    metadata_csv: str | Path,
    cif_dir: str | Path,
    require_cifs: bool = True,
    userccd_manifest_tsv: str | Path | None = None,
    userccd_path: str | Path | None = None,
) -> pd.DataFrame:
    sample_ids = read_sample_ids(sample_id_list)
    if not sample_ids:
        raise ValueError(f"Sample ID list is empty: {sample_id_list}")

    metadata_df = pd.read_csv(metadata_csv, keep_default_na=False)
    metadata_lookup = _build_metadata_lookup(metadata_df)
    userccd_lookup = (
        _build_userccd_lookup(userccd_manifest_tsv)
        if userccd_manifest_tsv is not None
        else None
    )
    cif_dir = Path(cif_dir)

    rows: list[dict[str, object]] = []
    missing_cifs: list[Path] = []
    target_cache: dict[str, TargetCcdMapping] = {}
    userccd_cache: dict[str, tuple[tuple[str, ...], str]] = {}
    for sample_id in sample_ids:
        parsed = parse_studio179_sample_id(sample_id)
        ccd_mapping = target_cache.get(parsed.disco_target)
        if ccd_mapping is None:
            ccd_mapping = resolve_target_ccd_mapping(parsed.disco_target, metadata_lookup)
            target_cache[parsed.disco_target] = ccd_mapping

        cif_path = cif_dir / f"{sample_id}.cif"
        cif_exists = cif_path.exists()
        if require_cifs and not cif_exists:
            missing_cifs.append(cif_path)

        row = {
            "sample_id": sample_id,
            "length": parsed.length,
            "disco_target": parsed.disco_target,
            "sample_index": parsed.sample_index,
            "ccd_codes": format_ccd_codes(ccd_mapping.ccd_codes),
            "target_signature": _target_signature_from_name(parsed.disco_target, metadata_lookup),
            "metadata_ligands": ";".join(ccd_mapping.metadata_ligands),
            "metadata_ligand_names": ";".join(ccd_mapping.metadata_ligand_names),
            "target_mapping_source": ccd_mapping.mapping_source,
            "cif_path": str(cif_path),
            "cif_exists": bool(cif_exists),
        }

        if userccd_lookup is not None:
            userccd_mapping = userccd_cache.get(parsed.disco_target)
            if userccd_mapping is None:
                userccd_mapping = resolve_target_userccd_mapping(
                    ccd_mapping,
                    userccd_lookup,
                )
                userccd_cache[parsed.disco_target] = userccd_mapping
            component_ids, mapping_source = userccd_mapping
            row["af3_ligand_ccd_codes"] = ";".join(component_ids)
            row["af3_user_ccd_path"] = str(userccd_path) if userccd_path else ""
            row["userccd_mapping_source"] = mapping_source

        rows.append(row)

    if missing_cifs:
        preview = "\n  ".join(str(path) for path in missing_cifs[:10])
        raise FileNotFoundError(
            f"Missing {len(missing_cifs)} CIF files referenced by {sample_id_list}. First paths:\n  {preview}"
        )

    return pd.DataFrame(rows)


def _component_token_lookup(manifest_df: pd.DataFrame) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for _, row in manifest_df.iterrows():
        target = str(row["disco_target"])
        signature = str(row["target_signature"])
        if "+" in target or "+" in signature:
            continue
        key = _canonical_component_key(target)
        if key and key not in lookup:
            lookup[key] = signature
    return lookup


def normalize_target_signatures(
    manifest_df: pd.DataFrame,
    targets: Iterable[str] | str | None,
) -> list[str]:
    target_values = _split_comma_values(targets)
    if not target_values:
        return []

    known_targets = {
        normalize_disco_key(row["disco_target"]): str(row["target_signature"])
        for _, row in manifest_df.iterrows()
    }
    component_lookup = _component_token_lookup(manifest_df)

    signatures: list[str] = []
    for target_value in target_values:
        target_key = normalize_disco_key(target_value)
        if target_key in known_targets:
            signature = known_targets[target_key]
        else:
            tokens: list[str] = []
            for component in [part for part in target_key.split("+") if part]:
                component_key = _canonical_component_key(component)
                tokens.append(component_lookup.get(component_key, component))
            signature = _format_target_signature(tokens)

        if signature and signature not in signatures:
            signatures.append(signature)

    return signatures


def filter_studio179_manifest(
    manifest_df: pd.DataFrame,
    *,
    targets: Iterable[str] | str | None = None,
    lengths: Iterable[int] | None = None,
    sample_indices: Iterable[int] | None = None,
) -> pd.DataFrame:
    selected_df = manifest_df.copy()

    requested_signatures = normalize_target_signatures(manifest_df, targets)
    if requested_signatures:
        selected_df = selected_df.loc[
            selected_df["target_signature"].isin(set(requested_signatures))
        ].copy()

    if lengths is not None:
        requested_lengths = set(lengths)
        selected_df = selected_df.loc[selected_df["length"].isin(requested_lengths)].copy()

    if sample_indices is not None:
        requested_indices = set(sample_indices)
        selected_df = selected_df.loc[selected_df["sample_index"].isin(requested_indices)].copy()

    return selected_df


def annotate_sample_dict_with_userccd(
    *,
    sample_dict: dict,
    selected_manifest_df: pd.DataFrame,
    userccd_path: str | Path,
) -> dict:
    if "af3_ligand_ccd_codes" not in selected_manifest_df.columns:
        raise ValueError("selected_manifest_df is missing af3_ligand_ccd_codes")

    manifest_by_sample_id = selected_manifest_df.set_index("sample_id", drop=False)
    for sample_id, sample_entry in sample_dict.items():
        if sample_id not in manifest_by_sample_id.index:
            raise ValueError(f"No Studio-179 manifest row for sample_id={sample_id}")
        row = manifest_by_sample_id.loc[sample_id]
        component_ids = split_semicolon_values(row["af3_ligand_ccd_codes"])

        pdb_chain_info = sample_entry["pdb_chain_info"]
        ligand_pn_unit_iids = list(pdb_chain_info.get("ligand_pn_unit_iids", []))
        if len(component_ids) != len(ligand_pn_unit_iids):
            raise ValueError(
                "Studio-179 userCCD component count does not match ligand chain count "
                f"for {sample_id}: {len(component_ids)} != {len(ligand_pn_unit_iids)}"
            )
        pdb_chain_info["af3_ligand_ccd_codes"] = component_ids
        pdb_chain_info["af3_user_ccd_path"] = str(userccd_path)

    return sample_dict
