from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


converter = pytest.importorskip(
    "allatom_design.eval.benchmarking.studio179.utils.convert_sdfs_to_userccd"
)
Chem = pytest.importorskip("rdkit.Chem")
AllChem = pytest.importorskip("rdkit.Chem.AllChem")


def _write_stereoany_sdf(path: Path) -> None:
    mol = Chem.AddHs(Chem.MolFromSmiles("FC=CF"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            bond.SetStereo(Chem.BondStereo.STEREOANY)
            bond.SetBondDir(Chem.BondDir.EITHERDOUBLE)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def test_sdf_to_user_ccd_normalizes_stereoany_and_adds_af3_required_keys(tmp_path: Path) -> None:
    sdf_path = tmp_path / "priority_1" / "test_ligand_final_0.sdf"
    sdf_path.parent.mkdir()
    _write_stereoany_sdf(sdf_path)

    cif_text, record = converter.sdf_to_user_ccd_text(
        sdf_path=sdf_path,
        component_id="S179001",
        ligand_name="test_ligand",
        include_hydrogens=False,
    )

    assert record.stereoany_bonds_normalized == 1
    assert "data_S179001" in cif_text
    assert "_chem_comp.type" in cif_text
    assert "non-polymer" in cif_text
    assert "_chem_comp.mon_nstd_parent_comp_id ?" in cif_text
    assert "_chem_comp_atom.pdbx_model_Cartn_x_ideal" in cif_text
    assert " S179001 " in cif_text
    converter.validate_user_ccd_text(cif_text, component_id="S179001")


def test_sdf_to_user_ccd_accepts_single_atom_metal_with_empty_bond_loop(tmp_path: Path) -> None:
    sdf_path = tmp_path / "priority_1" / "Cu_ideal.sdf"
    sdf_path.parent.mkdir()
    mol = Chem.RWMol()
    mol.AddAtom(Chem.Atom("Cu"))
    mol = mol.GetMol()
    conformer = Chem.Conformer(1)
    conformer.SetAtomPosition(0, (0.0, 0.0, 0.0))
    mol.AddConformer(conformer)
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol)
    writer.close()

    cif_text, record = converter.sdf_to_user_ccd_text(
        sdf_path=sdf_path,
        component_id="S179001",
        ligand_name="Cu",
        include_hydrogens=False,
    )

    assert record.num_atoms == 1
    assert "_chem_comp_bond.atom_id_1" in cif_text
    assert "_chem_comp_bond.value_order" in cif_text
    converter.validate_user_ccd_text(cif_text, component_id="S179001")


def test_build_conversion_jobs_uses_metadata_ligand_names_for_outputs(tmp_path: Path) -> None:
    sdf_root = tmp_path / "studio-179"
    (sdf_root / "priority_1").mkdir(parents=True)
    (sdf_root / "priority_2").mkdir(parents=True)
    (sdf_root / "priority_1" / "geosmin_final_0.sdf").write_text("")
    (sdf_root / "priority_2" / "Co_ideal.sdf").write_text("")
    metadata_csv = tmp_path / "all_diversity_results.csv"
    pd.DataFrame(
        [
            {
                "ligand": "geosmin_final_0",
                "ligand_name": "geosmin",
                "CCD": "",
                "priority": "1",
            },
        ]
    ).to_csv(metadata_csv, index=False)

    jobs = converter.build_conversion_jobs(
        sdf_root=sdf_root,
        metadata_csv=metadata_csv,
        output_dir=tmp_path / "conformer_cifs",
    )

    assert [job.component_id for job in jobs] == ["S179001", "S179002"]
    assert jobs[0].cif_path.name == "geosmin.cif"
    assert jobs[1].cif_path.name == "Co_ideal.cif"
