"""
Make positional constraint DataFrame for ligand pocket or scaffold regions.

Usage:
    python -m allatom_design.eval.sampling.make_pos_constraint_df

This script reads CIF files, annotates ligand pockets, and creates a DataFrame
with positional constraints in the format "A1-10,B5-8" for either:
- pocket regions (residues within pocket_distance of ligands)
- scaffold regions (residues NOT within pocket_distance of ligands)
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig
import hydra
from allatom_design.eval.utils.constraint_utils import create_pos_constraint_dict_from_pocket
from allatom_design.eval.utils.eval_setup_utils import get_pdb_files
from allatom_design.eval.utils.misc import _parallel_context

def make_pos_constraint_df(
    pdb_cfg: DictConfig = None,
    sampling_inputs_df: pd.DataFrame = None,
    output_path: str = None,
    pocket_distance: float = 5.0,
    constraint_type: str = "pocket",  # "pocket" or "scaffold"
    cif_parse_cfg: DictConfig = None,
    preprocess_cfg: DictConfig = None,
    sample_is_designed: bool = False,
    debug: bool = False,
    num_debug_samples: int = 5,
    save_ligand_mpnn_csv: bool = True,
    use_calpha_for_pocket_annotation: bool = False,
    num_workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a positional constraint DataFrame for multiple CIF files.

    Args:
        cif_dir: Directory containing CIF files
        pdb_list_file: Text file with list of CIF filenames (one per line). If None, use all CIFs in cif_dir.
        output_path: Path to save the output parquet file
        pocket_distance: Distance threshold for pocket identification
        constraint_type: "pocket" or "scaffold"
        data_cfg: Configuration for CIF parsing
        transform_cfg: Configuration for preprocessing and featurization
        pdb_chain_info_dict: Dictionary mapping pdb_id to pdb_chain_info
        debug: If True, only process num_debug_samples samples
        num_debug_samples: Number of samples to process in debug mode
        save_ligand_mpnn_csv: If True, also save LigandMPNN input CSV

    Returns:
        Tuple of (positional constraint DataFrame, LigandMPNN input DataFrame)
    """
    # Get list of CIF files to process
    sample_paths = get_pdb_files(**pdb_cfg)
    pdb_ids = [Path(sample_path).stem for sample_path in sample_paths]
    if sampling_inputs_df is not None:
        pdb_id_set = set(sampling_inputs_df["pdb_id"].values)
        valid_indices = [i for i, pdb_id in enumerate(pdb_ids) if pdb_id in pdb_id_set]
        sample_paths = [sample_paths[i] for i in valid_indices]
        pdb_ids = [pdb_ids[i] for i in valid_indices]

    # Debug mode: limit number of samples
    if debug:
        sample_paths = sample_paths[:num_debug_samples]
        pdb_ids = pdb_ids[:num_debug_samples]
        print(f"[DEBUG MODE] Processing only {len(sample_paths)} samples")

    print(f"Found {len(sample_paths)} samples to process")

    rows = []
    failed_pdbs = []
    results_for_ligand_mpnn = []

    if num_workers > 1:
        if Parallel is None:
            raise ImportError("joblib is required when num_workers > 1") from None
        print(f"Using {num_workers} workers for parallel processing")
        results = Parallel(n_jobs=num_workers, backend="loky")(
            delayed(_make_single_pos_constraint_dict)(
                sample_path=sample_path,
                sampling_inputs_df=sampling_inputs_df,
                cif_parse_cfg=cif_parse_cfg,
                preprocess_cfg=preprocess_cfg,
                sample_is_designed=sample_is_designed,
                pocket_distance=pocket_distance,
                constraint_type=constraint_type,
                save_ligand_mpnn_csv=save_ligand_mpnn_csv,
                use_calpha_for_pocket_annotation=use_calpha_for_pocket_annotation,
            )
            for sample_path in tqdm(sample_paths, desc=f"Dispatching samples ({constraint_type})")
        )
        for result in results:
            if result["status"] == "ok":
                rows.append(result["pos_constraint_dict"])
                if result["ligand_mpnn_dict"]:
                    results_for_ligand_mpnn.append(result["ligand_mpnn_dict"])
            elif result["status"] == "no_ligand":
                print(f"Warning: No ligand found in {result['pdb_key']}, skipping...")
                failed_pdbs.append(result["pdb_key"])
            else:
                print(f"Error processing {result['pdb_key']}: {result.get('error_msg', 'unknown')}")
                failed_pdbs.append(result["pdb_key"])
    else:
        for sample_path in tqdm(sample_paths, desc=f"Processing samples ({constraint_type})"):
            result = _make_single_pos_constraint_dict(
                sample_path=sample_path,
                sampling_inputs_df=sampling_inputs_df,
                cif_parse_cfg=cif_parse_cfg,
                preprocess_cfg=preprocess_cfg,
                sample_is_designed=sample_is_designed,
                pocket_distance=pocket_distance,
                constraint_type=constraint_type,
                save_ligand_mpnn_csv=save_ligand_mpnn_csv,
                use_calpha_for_pocket_annotation=use_calpha_for_pocket_annotation,
            )
            if result["status"] == "ok":
                rows.append(result["pos_constraint_dict"])
                if result["ligand_mpnn_dict"]:
                    results_for_ligand_mpnn.append(result["ligand_mpnn_dict"])
            elif result["status"] == "no_ligand":
                print(f"Warning: No ligand found in {result['pdb_key']}, skipping...")
                failed_pdbs.append(result["pdb_key"])
            else:
                print(f"Error processing {result['pdb_key']}: {result.get('error_msg', 'unknown')}")
                failed_pdbs.append(result["pdb_key"])

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Create LigandMPNN DataFrame
    ligand_mpnn_input_df = pd.DataFrame(results_for_ligand_mpnn) if results_for_ligand_mpnn else pd.DataFrame()

    print(f"\nSuccessfully processed {len(df)} CIF files")
    print(f"Failed: {len(failed_pdbs)} CIF files")

    if failed_pdbs:
        print(f"Failed PDBs: {failed_pdbs[:10]}{'...' if len(failed_pdbs) > 10 else ''}")

    # Save to file if output_path is provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Drop metadata columns before saving (for minimal version)
        cols_to_drop = ["pocket_distance", "constraint_type", "num_constrained_residues"]
        df_to_save = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        if output_path.suffix == ".parquet":
            df_to_save.to_parquet(output_path)
        elif output_path.suffix == ".csv":
            df_to_save.to_csv(output_path)
        else:
            # Default to csv
            df_to_save.to_csv(output_path)

        print(f"Saved positional constraint DataFrame to {output_path}")

        # Also save full version with metadata
        full_output_path = output_path.parent / (output_path.stem + "_full" + output_path.suffix)
        if full_output_path.suffix == ".parquet":
            df.to_parquet(full_output_path)
        elif full_output_path.suffix == ".csv":
            df.to_csv(full_output_path)
        else:
            df.to_parquet(full_output_path)

        print(f"Saved full positional constraint DataFrame to {full_output_path}")

        # Save LigandMPNN input CSV
        if save_ligand_mpnn_csv and len(ligand_mpnn_input_df) > 0:
            ligand_mpnn_csv_path = output_path.parent / (output_path.stem + "_for_ligandmpnn.csv")
            # Select only required columns for LigandMPNN: pdb_path, chains, fixed_residues
            ligand_mpnn_df_to_save = ligand_mpnn_input_df[['pdb_path', 'chains', 'fixed_residues']]
            ligand_mpnn_df_to_save.to_csv(ligand_mpnn_csv_path, index=False)
            print(f"Saved LigandMPNN input CSV to {ligand_mpnn_csv_path}")

    return df, ligand_mpnn_input_df


@hydra.main(config_path="../../configs/eval/sampling", config_name="make_pos_constraint_df", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Create positional constraint DataFrame for ligand pocket or scaffold regions.
    """
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.sampling_inputs_csv is not None:
        sampling_inputs_df = pd.read_csv(cfg.sampling_inputs_csv)
    else:
        sampling_inputs_df = None

    if not cfg.source_is_designed:
        cif_parse_cfg = cfg.cif_cfg.parse.native
        preprocess_cfg = cfg.preprocess_cfg.native
    else:
        cif_parse_cfg = cfg.cif_cfg.parse.designed_samples
        preprocess_cfg = cfg.preprocess_cfg.designed_samples

    # Determine constraint types to process
    if cfg.constraint_type == "both":
        constraint_types = ["pocket", "scaffold"]
    else:
        constraint_types = [cfg.constraint_type]

    for constraint_type in constraint_types:
        print(f"\n{'='*60}")
        print(f"Creating {constraint_type} positional constraint DataFrame")
        print(f"Pocket distance: {cfg.pocket_distance} Å")
        print(f"{'='*60}\n")

        if not cfg.debug:
            output_filename = f"pos_constraint_{constraint_type}_{cfg.pocket_distance}A.csv"
        else:
            output_filename = f"debug_pos_constraint_{constraint_type}_{cfg.pocket_distance}A.csv"
        output_path = output_dir / output_filename

        df, ligand_mpnn_df = make_pos_constraint_df(
            pdb_cfg=cfg.pdb_cfg,
            sampling_inputs_df=sampling_inputs_df,
            output_path=str(output_path),
            pocket_distance=cfg.pocket_distance,
            constraint_type=constraint_type,
            cif_parse_cfg=cif_parse_cfg,
            preprocess_cfg=preprocess_cfg,
            sample_is_designed=cfg.get("source_is_designed", False),
            debug=cfg.get("debug", False),
            num_debug_samples=cfg.get("num_debug_samples", 5),
            save_ligand_mpnn_csv=cfg.get("save_ligand_mpnn_csv", True),
            use_calpha_for_pocket_annotation=cfg.get("use_calpha_for_pocket_annotation", False),
            num_workers=cfg.get("num_workers", 1),
        )

        # Print summary statistics
        if len(df) > 0:
            print(f"\nSummary for {constraint_type}:")
            print(f"  Total entries: {len(df)}")
            print(f"  Entries with constraints: {(df['num_constrained_residues'] > 0).sum()}")
            print(f"  Average constrained residues: {df['num_constrained_residues'].mean():.1f}")
            print(f"  Max constrained residues: {df['num_constrained_residues'].max()}")
            print(f"\nSample entries:")
            print(df.head())

        if len(ligand_mpnn_df) > 0:
            print(f"\nLigandMPNN input CSV summary:")
            print(f"  Total entries: {len(ligand_mpnn_df)}")
            print(f"  Entries with fixed_residues: {(ligand_mpnn_df['fixed_residues'] != '').sum()}")
            print(f"\nSample LigandMPNN entries:")
            print(ligand_mpnn_df[['pdb_path', 'chains', 'fixed_residues']].head())

    print(f"\n{'='*60}")
    print("Done!")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
