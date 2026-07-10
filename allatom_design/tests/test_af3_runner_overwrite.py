import sys
import types

from omegaconf import OmegaConf

from allatom_design.eval.structure_prediction.af3_runner import (
    _af3_prediction_outputs_complete,
    _af3_overwrite_enabled,
    _load_af3_runner,
    _prepare_af3_prediction_run,
    _run_af3_inprocess,
    _write_af3_input_fingerprint,
    summarize_af3_prediction_outputs,
)


def test_af3_overwrite_enabled_parses_config_values() -> None:
    assert _af3_overwrite_enabled(OmegaConf.create({"overwrite": "true"}))
    assert not _af3_overwrite_enabled(OmegaConf.create({}))


def test_load_af3_runner_cache_is_keyed_by_resolved_path(monkeypatch, tmp_path) -> None:
    runner_a = tmp_path / "runner_a.py"
    runner_b = tmp_path / "runner_b.py"
    runner_a.write_text("MARKER = 'a'\n")
    runner_b.write_text("MARKER = 'b'\n")
    monkeypatch.setattr(
        "allatom_design.eval.structure_prediction.af3_runner._AF3_RUNNER_MOD",
        None,
    )
    monkeypatch.setattr(
        "allatom_design.eval.structure_prediction.af3_runner._AF3_RUNNER_PATH",
        None,
    )

    loaded_a = _load_af3_runner(str(runner_a))
    loaded_b = _load_af3_runner(str(runner_b))

    assert loaded_a.MARKER == "a"
    assert loaded_b.MARKER == "b"
    assert loaded_a is not loaded_b


def test_run_af3_inprocess_uses_selected_mode_config(monkeypatch, tmp_path) -> None:
    json_path = tmp_path / "sample_a.json"
    json_path.write_text('{"modelSeeds": [1]}')

    class FakeFoldInput:
        def sanitised_name(self) -> str:
            return "sample_a"

    folding_input_module = types.ModuleType("alphafold3.common.folding_input")
    folding_input_module.load_fold_inputs_from_path = lambda _: [FakeFoldInput()]

    common_module = types.ModuleType("alphafold3.common")
    common_module.folding_input = folding_input_module
    alphafold3_module = types.ModuleType("alphafold3")
    alphafold3_module.common = common_module

    monkeypatch.setitem(sys.modules, "alphafold3", alphafold3_module)
    monkeypatch.setitem(sys.modules, "alphafold3.common", common_module)
    monkeypatch.setitem(
        sys.modules,
        "alphafold3.common.folding_input",
        folding_input_module,
    )

    captured_kwargs = {}

    class FakeRunner:
        def process_fold_input(self, **kwargs) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(
        "allatom_design.eval.structure_prediction.af3_runner._get_af3_model_runner_and_config",
        lambda **_: (FakeRunner(), object(), object()),
    )

    _run_af3_inprocess(
        json_path=str(json_path),
        out_dir=str(tmp_path / "preds"),
        runner_path="/fake/run_alphafold.py",
        inference_config=OmegaConf.create(
            {
                "base": {},
                "ss": {
                    "num_diffusion_samples": 1,
                    "max_templates": 2,
                    "ligand_protein_template_conditioning_mode": 3,
                    "fix_standalone_glycans": False,
                },
            }
        ),
        mode="ss",
    )

    assert captured_kwargs["max_templates"] == 2
    assert captured_kwargs["ligand_protein_template_conditioning_mode"] == 3
    assert captured_kwargs["fix_standalone_glycans"] is False


def test_strict_af3_reuse_rejects_count_complete_stale_prediction(tmp_path) -> None:
    json_path = tmp_path / "sample_a.json"
    json_path.write_text('{"name": "sample_a", "modelSeeds": [1]}')
    pred_dir = tmp_path / "preds" / "sample_a" / "seed-1"
    pred_dir.mkdir(parents=True)
    (pred_dir / "sample_a_model.cif").write_text("data_sample_a\n")
    inference_config = OmegaConf.create(
        {
            "base": {},
            "ss": {
                "num_diffusion_samples": 1,
                "strict_input_fingerprint": True,
            },
        }
    )
    _write_af3_input_fingerprint(
        json_path=json_path,
        out_dir=tmp_path / "preds",
        inference_config=inference_config,
        mode="ss",
    )

    assert _af3_prediction_outputs_complete(
        tmp_path / "preds",
        "sample_a",
        1,
        json_path=json_path,
        inference_config=inference_config,
        mode="ss",
        strict_input_fingerprint=True,
    )

    json_path.write_text('{"name": "sample_a", "modelSeeds": [2]}')
    summary = summarize_af3_prediction_outputs(
        out_dir=tmp_path / "preds",
        job_name="sample_a",
        expected_count=1,
        json_path=json_path,
        inference_config=inference_config,
        mode="ss",
        strict_input_fingerprint=True,
    )

    assert summary["n_found"] == 1
    assert summary["input_fingerprint_ok"] is False
    assert not summary["complete"]


def test_strict_af3_prepare_removes_stale_prediction_dir(tmp_path) -> None:
    json_path = tmp_path / "sample_a.json"
    json_path.write_text('{"name": "sample_a", "modelSeeds": [1]}')
    sample_root = tmp_path / "preds" / "sample_a"
    pred_dir = sample_root / "seed-1"
    pred_dir.mkdir(parents=True)
    (pred_dir / "sample_a_model.cif").write_text("data_sample_a\n")
    inference_config = OmegaConf.create(
        {
            "base": {},
            "ss": {
                "num_diffusion_samples": 1,
                "strict_input_fingerprint": True,
            },
        }
    )

    should_reuse = _prepare_af3_prediction_run(
        json_path=str(json_path),
        out_dir=str(tmp_path / "preds"),
        inference_config=inference_config,
        mode="ss",
    )

    assert not should_reuse
    assert not sample_root.exists()


def test_af3_summary_requires_exact_prediction_count(tmp_path) -> None:
    json_path = tmp_path / "sample_a.json"
    json_path.write_text('{"name": "sample_a", "modelSeeds": [1]}')
    for idx in range(2):
        pred_dir = tmp_path / "preds" / "sample_a" / f"seed-{idx}"
        pred_dir.mkdir(parents=True)
        (pred_dir / f"sample_a_{idx}_model.cif").write_text("data_sample_a\n")

    summary = summarize_af3_prediction_outputs(
        out_dir=tmp_path / "preds",
        job_name="sample_a",
        expected_count=1,
    )

    assert summary["n_found"] == 2
    assert summary["n_surplus"] == 1
    assert not summary["complete"]
