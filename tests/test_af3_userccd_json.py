from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from biotite.structure import AtomArray

from allatom_design.eval.eval_utils.folding_utils import make_af3_json


def _make_protein_metal_atom_array() -> AtomArray:
    records = [
        ("A", 1, "ALA", False, "N", "N", [0.0, 0.0, 0.0], "A_1"),
        ("A", 1, "ALA", False, "CA", "C", [1.0, 0.0, 0.0], "A_1"),
        ("A", 1, "ALA", False, "C", "C", [2.0, 0.0, 0.0], "A_1"),
        ("A", 1, "ALA", False, "O", "O", [3.0, 0.0, 0.0], "A_1"),
        ("B", 1, "l01", True, "CU1", "CU", [1.0, 2.0, 3.0], "B_1"),
    ]
    atom_array = AtomArray(len(records))
    atom_array.chain_id = np.array([record[0] for record in records])
    atom_array.res_id = np.array([record[1] for record in records])
    atom_array.res_name = np.array([record[2] for record in records])
    atom_array.hetero = np.array([record[3] for record in records], dtype=bool)
    atom_array.atom_name = np.array([record[4] for record in records])
    atom_array.element = np.array([record[5] for record in records])
    atom_array.coord = np.array([record[6] for record in records], dtype=np.float32)
    atom_array.set_annotation("pn_unit_iid", np.array([record[7] for record in records]))
    return atom_array


def test_make_af3_json_uses_userccd_component_ids_without_metal_rewrite(tmp_path: Path) -> None:
    sample_dict = {
        "length_150_Cu_sample_0": {
            "designed_sample_id": ["length_150_Cu_sample_0_sample0"],
            "designed_sample_atom_array": [_make_protein_metal_atom_array()],
            "pdb_chain_info": {
                "protein_pn_unit_iids": ["A_1"],
                "ligand_pn_unit_iids": ["B_1"],
                "ligand_ccd_codes": ["l01"],
                "af3_ligand_ccd_codes": ["S179002"],
            },
        }
    }

    make_af3_json(
        af3_ss_input_dir=tmp_path,
        sample_dict=sample_dict,
        json_config={
            "model_seeds": [42],
            "version": 4,
            "user_ccd_path": "/tmp/studio179_all_components_userccd.cif",
        },
    )

    payload = json.loads((tmp_path / "length_150_Cu_sample_0_sample0.json").read_text())
    ligand_entries = [entry["ligand"] for entry in payload["sequences"] if "ligand" in entry]

    assert payload["version"] == 4
    assert payload["userCCDPath"] == "/tmp/studio179_all_components_userccd.cif"
    assert ligand_entries == [{"id": "B", "ccdCodes": ["S179002"]}]
