import pickle, os, yaml
from dataclasses import asdict
from boltz.model.model import Boltz1
from boltz.main import BoltzDiffusionParams
import time
import argparse
from utils import motif_utils, mydesign_utils
from designer import MultistateDesigner
from losses import (
    MotifLoss, AntiMotifLoss, ContactLoss, HelixBiasLoss,
    ConfChangeLoss, LigandContactLoss, AntiLigandContactLoss,
    SheetBiasLoss, SequenceSimilarityLoss, RadiusOfGyrationLoss,
)

device = "cuda"


def _init_boltz(recycles=0):
    predict_args = {
        "recycling_steps": recycles,
        "sampling_steps": 200,
        "diffusion_samples": 1,
        "write_confidence_summary": True,
        "write_full_pae": True,
        "write_full_pde": True,
    }
    diffusion_params = BoltzDiffusionParams()
    diffusion_params.step_scale = 1.638
    return Boltz1.load_from_checkpoint(
        "boltz/boltz1_conf.ckpt",
        strict=False,
        predict_args=predict_args,
        map_location=device,
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        structure_prediction_training=True,
        no_msa=False,
        no_atom_encoder=False,
    ).eval().requires_grad_(False)


def _build_loss(loss_cfg, motif_templates):
    t = loss_cfg["type"]
    if t in ("MotifLoss", "AntiMotifLoss"):
        motif = motif_templates[loss_cfg["motif"]]
        return MotifLoss(motif) if t == "MotifLoss" else AntiMotifLoss(motif)
    if t == "ConfChangeLoss":
        return ConfChangeLoss(strength=loss_cfg.get("strength", 1.0))
    if t in ("LigandContactLoss", "AntiLigandContactLoss"):
        cls = LigandContactLoss if t == "LigandContactLoss" else AntiLigandContactLoss
        return cls(idx=loss_cfg.get("idx"), strength=loss_cfg.get("strength", 1.0))
    if t == "HelixBiasLoss":
        return HelixBiasLoss(strength=loss_cfg.get("strength", 0.0))
    if t == "SheetBiasLoss":
        return SheetBiasLoss(strength=loss_cfg.get("strength", 0.0))
    if t == "RadiusOfGyrationLoss":
        return RadiusOfGyrationLoss(strength=loss_cfg.get("strength", 1.0))
    if t == "SequenceSimilarityLoss":
        return SequenceSimilarityLoss(loss_cfg["target_sequence"], strength=loss_cfg.get("strength", 1.0))
    if t == "ContactLoss":
        return ContactLoss()
    raise ValueError(f"Unknown loss type: {t!r}")


def build_designer(config, length_override=None, visualize=False):
    num_states = config["num_states"]
    designer = MultistateDesigner(num_states=num_states, visualize=visualize)

    motif_names = config.get("motifs", [])
    if motif_names:
        motif_templates = motif_utils.get_motif_scaffold_templates(
            paths=[f"motifs/{m}.pdb" for m in motif_names],
            target_length=length_override or config.get("length"),
        )
        length = len(motif_templates[0]["motif_mask"])
    else:
        motif_templates = []
        length = length_override or config.get("length")
        if length is None:
            raise ValueError("No motifs specified and no length set — provide 'length' in the config or pass --length")

    for motif in motif_templates:
        designer.add_motif(motif)

    for i, state_cfg in enumerate(config.get("states", [])):
        if i >= num_states:
            break
        for lig_str in (state_cfg or []):
            mol_type, value = lig_str.split(":", 1)
            designer.add_ligand((value, mol_type), state=i)

    for loss_cfg in config.get("losses", []):
        loss = _build_loss(loss_cfg, motif_templates)
        designer.add_loss(loss, state=loss_cfg["state"], weight=loss_cfg.get("weight", 1.0))

    contact_loss = config.get("contact_loss", True)
    designer.initialize(length=length, contact_loss=contact_loss)
    return designer, motif_names, motif_templates


def run(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    motif_names = config.get("motifs", [])
    out_dir = os.path.join(args.outpath, "_".join(motif_names)) if motif_names else args.outpath
    os.makedirs(out_dir, exist_ok=True)

    config_name = os.path.splitext(os.path.basename(args.config))[0]

    for design in range(args.worker_id, args.num_designs, args.num_workers):
        print(f"\nStarting {config_name} design {design+1}/{args.num_designs}")

        boltz_model = _init_boltz(args.recycles)
        designer, motif_names, motif_templates = build_designer(
            config, length_override=args.length, visualize=args.visualize
        )

        t0 = time.perf_counter()
        print("Optimizing sequence...")
        designer.optimize(boltz_model, verbose=args.verbose, debug=args.debug)
        print(f"Optimization done in {time.perf_counter() - t0:.1f} sec")

        print("Saving structures...")
        t2 = time.perf_counter()
        structs = designer.get_final_structs(boltz_model)
        print(f"Structure generation took {time.perf_counter() - t2:.1f} sec")

        design_dir = os.path.join(out_dir, f"design{design}")
        os.makedirs(design_dir, exist_ok=True)

        for i, motif_name in enumerate(motif_names):
            with open(os.path.join(design_dir, f"{motif_name}_spec.pkl"), "wb") as f:
                pickle.dump(motif_templates[i], f)

        mydesign_utils.save_structs(structs, design_dir)

        if args.ligandmpnn_seqs > 0:
            designer.do_lmpnn_redesign(boltz_model, design_dir, structs, num_seqs=args.ligandmpnn_seqs)

        print(f"Finished design {design+1} in {time.perf_counter() - t0:.1f} sec total")

        if args.visualize:
            designer.save_visualization_info(design_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML design config (see tasks/)")
    parser.add_argument("--num_designs", type=int, default=1)
    parser.add_argument("-o", "--outpath", type=str, default="./out/")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--recycles", default=0, type=int)
    parser.add_argument("--length", default=None, type=int, help="Override design length from config")
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--worker_id", default=0, type=int)
    parser.add_argument("--ligandmpnn_seqs", default=0, type=int)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    run(args)
