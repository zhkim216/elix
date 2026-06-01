from pathlib import Path

from allatom_design.eval.eval_utils.eval_setup_utils import get_pdb_files


def test_get_pdb_files_filters_compact_and_legacy_sample_names(tmp_path: Path) -> None:
    names = [
        "CA_len150_0.cif",
        "CA_len150_4.cif",
        "CA_len150_5.cif",
        "CA_len250_0.cif",
        "legacy_len_150_4_model_0.cif",
        "legacy_len_250_0_model_0.cif",
    ]
    for name in names:
        (tmp_path / name).write_text("data_test\n")

    selected = get_pdb_files(
        pdb_dir=str(tmp_path),
        pdb_name_list=None,
        pdb_name_ext=".cif",
        sample_indices=[0, 4],
        sample_lengths=[150],
    )

    assert [Path(path).name for path in selected] == [
        "CA_len150_0.cif",
        "CA_len150_4.cif",
        "legacy_len_150_4_model_0.cif",
    ]
