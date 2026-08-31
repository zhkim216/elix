from boltz.data.parse.schema import parse_boltz_schema
import torch
import copy
import numpy as np
import pickle
from typing import Optional
from utils.mydesign_utils import get_batch, run_model, Annealer, norm_seq_grad
import os,time
from utils.tied_lmpnn import perform_tied_lmpnn_redesign
from utils import mydesign_utils
from losses import *
from pathlib import Path

torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32= os.getenv("BOLTZ_USE_CUEQ", "1").lower() in ("1", "true", "yes", "on") # for cueq kernels



with open(Path(__file__).resolve().parent / "boltz/ccd.pkl", "rb") as f:
    CCD_LIB = pickle.load(f)


def revcomp(seq: str) -> str:
    complement = str.maketrans("ACGT", "TGCA")
    return seq.upper().translate(complement)[::-1]
    
def get_batch_with_ligands(seq, ligands=None, device="cuda"):
    data = {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": ["A"],
                    "sequence": seq,
                    "msa": "empty",
                }
            },
        ],
    }
    ALPHABET = "BCDEFGHIJKLMNOPQRSTUVWXYZ"
    for ligand in ligands:
        if ligand is not None:
            assert isinstance(ligand, tuple) and len(ligand) == 2, "ligand must be a (value, mol_type) tuple"
            ligand, mol_type = ligand
            if mol_type == "ligand":
                data["sequences"].append({
                    "ligand": {
                        "id": [ALPHABET[0]],
                        "smiles": ligand,
                    }
                })
                ALPHABET = ALPHABET[1:]
            elif mol_type == "ccd":
                data["sequences"].append({
                    "ligand": {
                        "id": [ALPHABET[0]],
                        "ccd": ligand,
                    }
                })
                ALPHABET = ALPHABET[1:]
            elif mol_type == "protein":
                data["sequences"].append({
                    "protein": {
                        "id": [ALPHABET[0]],
                        "sequence": ligand,
                        "msa": "empty",
                    }
                })
                ALPHABET = ALPHABET[1:]
            elif mol_type in "rna":
                data["sequences"].append({
                    mol_type: {
                        "id": [ALPHABET[0]],
                        "sequence": ligand,
                    }
                })
                ALPHABET = ALPHABET[1:]
            elif mol_type in "dna":
                data["sequences"].append({
                    "dna": {
                        "id": [ALPHABET[0]],
                        "sequence": ligand,
                    }
                })
                data["sequences"].append({
                    "dna": {
                        "id": [ALPHABET[1]],
                        "sequence": revcomp(ligand),
                    }
                })
                ALPHABET = ALPHABET[2:]
            else:
                raise ValueError(f"Unsupported mol_type: {mol_type!r}. Valid types: ligand, ccd, protein, dna, rna")
        else:
            raise ValueError(f"Unsupported mol_type: {mol_type}")

    target = parse_boltz_schema(None, data, CCD_LIB)
    batch, structure = get_batch(target)
    batch = {key: value.unsqueeze(0).to(device) for key, value in batch.items()}
    # batch["msa"] = batch["res_type_logits"].unsqueeze(0).to(device)
    batch["msa_paired"] = torch.ones(
        batch["res_type"].shape[0], 1, batch["res_type"].shape[1]
    ).to(device)
    batch["deletion_value"] = torch.zeros(
        batch["res_type"].shape[0], 1, batch["res_type"].shape[1]
    ).to(device)
    batch["has_deletion"] = torch.full(
        (batch["res_type"].shape[0], 1, batch["res_type"].shape[1]), False
    ).to(device)
    batch["msa_mask"] = torch.ones(
        batch["res_type"].shape[0], 1, batch["res_type"].shape[1]
    ).to(device)
    batch["profile"] = batch["msa"].float().mean(dim=0).to(device)
    batch["deletion_mean"] = torch.zeros(batch["deletion_mean"].shape).to(device)
    batch["res_type"] = batch["res_type"].float()

    return batch, structure



        
class MultistateDesigner:
    def __init__(self, num_states=1, radius_gyr=False, visualize=False):
        self.ligands = [[] for _ in range(num_states)]
        self.motifs = []
        self.losses = []
        self.loss_log = []
        self.radius_gyr = radius_gyr
        
        self.visualize = visualize
        self.pseudo_logit_traj = []
        
        
    def add_ligand(self, ligand, state):
        if type(ligand) is list:
            self.ligands[state].extend(ligand)
        else:
            self.ligands[state].append(ligand)
            
    def add_motif(self, motif):
        self.motifs.append(motif)

    def add_loss(self, loss, state, weight=1.0):
        self.losses.append((loss, state, weight))
        
    def initialize(self, length, device='cuda', seq=None, seq_weight=0.0, contact_loss=True):
        
        self.device = device
        alphabet = list("XXARNDCQEGHILKMFPSTWYV-")

        z = torch.distributions.Gumbel(0, 1).sample((length, 33)).to(device)
        z[...,:2] = z[...,22:] = -np.inf
        self.logits = z.softmax(-1)
        
        ### build the motif mask ###
        self.fixed_mask = torch.zeros(length, dtype=bool, device=device)
        self.fixed_aa = torch.zeros_like(self.logits)

        start_seq = ['X']*length
        
        for i, motif in enumerate(self.motifs):
            if motif is not None:
                motif_mask = torch.from_numpy(motif['motif_mask']).to(device)
                
                self.fixed_mask |= motif_mask
                motif_seq = [alphabet.index(c) for c in motif['motif_seq']]     
                motif_seq = torch.nn.functional.one_hot(
                    torch.tensor(motif_seq), num_classes=22
                )
                self.fixed_aa[motif_mask,:22] = motif_seq.to(device)[motif_mask].float()
                for j in range(length):
                    if motif['motif_mask'][j]:
                        start_seq[j] = motif['motif_seq'][j]
                
        self.logits = torch.where(
            self.fixed_mask[...,None], 
            self.fixed_aa,
            self.logits
        )

        ## make the boltz batch objects
        self.batches = []
        for i, ligs in enumerate(self.ligands):
            self.batches.append(get_batch_with_ligands(''.join(start_seq), ligs, device)[0])
            if contact_loss:
                self.add_loss(ContactLoss(), state=i)
            if self.radius_gyr:
                self.add_loss(RadiusOfGyrationLoss(), state=i)

        

    def get_seq(self):
        alphabet = list("XXARNDCQEGHILKMFPSTWYV-")
        logits = self.logits.clone()
        invalid_idx = [0, 1, 6, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        logits[..., invalid_idx] = -float("inf")
        return "".join(alphabet[i.item()] for i in logits.argmax(-1))


    def get_final_structs(self, boltz_model, samples: Optional[int] = 5, set_seq: Optional[str] = None):
        predict_args={
            "recycling_steps": 3,
            "sampling_steps": 200,
            "diffusion_samples": samples,
            "write_confidence_summary": True,
            "write_full_pae": True,
            "write_full_pde": True,
        }
        results = []
        for i, ligands in enumerate(self.ligands):
            seq = self.get_seq() if set_seq is None else set_seq
            new_batch, new_struct = get_batch_with_ligands(seq, ligands)

            output = run_model(boltz_model, new_batch, predict_args)
            coords_all = output["coords"]

            struct_list = []
            for j in range(coords_all.shape[0]):
                struct_copy = copy.deepcopy(new_struct)
                struct_copy.atoms["coords"] = (
                    coords_all[j, : len(new_struct.atoms)].cpu().numpy()
                )
                struct_list.append(struct_copy)

            results.append((output, struct_list, i))
            
        return results
        
    def get_restype_from_logits(self, res_type_logits, opt, alpha=2.0):
        device = res_type_logits.device
        logits = alpha * res_type_logits
        
        X = logits - torch.sum(
            torch.eye(logits.shape[-1])[
                [0, 1, 6, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            ],
            dim=0,
        ).to(device) * (1e10)
        soft = torch.softmax(X / opt["temp"], dim=-1)  # probs
        hard = torch.zeros_like(soft).scatter_(
            -1, soft.max(dim=-1, keepdim=True)[1], 1.0
        )  # one hot
        hard = (hard - soft).detach() + soft # carries same grad

        pseudo = (
            opt["soft"] * soft + (1 - opt["soft"]) * res_type_logits
        )  # interp between probs and logits
        pseudo = (
            opt["hard"] * hard + (1 - opt["hard"]) * pseudo
        )  # interp between on hot and the above
    
        return {'soft': soft, 'hard': hard, 'pseudo': pseudo}

    def get_loss(self, restype, boltz_model, opt, verbose=False):
        total_loss = 0
        loss_dict = []
    
        boltz_out = []
        # prep for boltz
        for batch in self.batches:
            batch['res_type'] = torch.cat([
                restype['pseudo'][None],
                batch['res_type'][:,len(restype['pseudo']):].detach()
            ], 1)
            batch["msa"] = batch["res_type"].unsqueeze(0).detach()
            batch["profile"] = batch["msa"].float().mean(dim=0).detach()

            dict_out = boltz_model.get_distogram(batch)[0]
            boltz_out.append(batch | dict_out | {'restype': restype})


        for loss, state, weight in self.losses:
            if type(state) is list:
                readout = [boltz_out[s] for s in state]
            else:
                readout = boltz_out[state]
            this_loss = loss.evaluate(readout, boltz_model.device, opt)
            
            loss_dict.append((type(loss).__name__, state, this_loss.item()))
            total_loss = total_loss + weight * this_loss

        if verbose:
            print(loss_dict)
            print(self.get_seq())
        
        self.loss_log.append(loss_dict)
        
        return total_loss
        
            
    def do_iter(self, boltz_model, opt, pre_run=False, verbose=False):

        self.logits.requires_grad = True
        restype = self.get_restype_from_logits(self.logits, opt)
        restype = {
            k: torch.where(self.fixed_mask[...,None], self.fixed_aa, v)\
            for k, v in restype.items()
        }
        
        if self.visualize:
            with torch.no_grad():
                self.pseudo_logit_traj.append(restype["pseudo"].detach().cpu().clone())
       
        loss = self.get_loss(restype, boltz_model, opt, verbose=verbose)
        # loss = restype.sum()
    
        loss.backward()
        if verbose: print('total_loss', loss)
        
        
        with torch.no_grad():
            self.logits.grad[self.fixed_mask] = 0
            self.logits.grad[
                ..., [0, 1, 6, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            ] = 0
            self.logits.grad = norm_seq_grad(
                self.logits.grad[None], 
                torch.ones_like(self.logits[:,0])
            )[0]
        
            self.logits -= opt["lr_rate"] * self.logits.grad
        self.logits.grad = None
        
                

    def optimize(self, boltz_model, verbose=False, debug=False):
        print(f"stage 1: warmup")

        for opt in Annealer(hard=0, e_hard=0, iters=30, lr=0.2):
            self.do_iter(boltz_model, opt, pre_run=True, verbose=verbose)
        if debug: return
        
        with torch.no_grad():
            self.logits = self.get_restype_from_logits(self.logits, opt)['pseudo']
    
        print(f"stage 2: exploration")
        for opt in Annealer(
            soft=0, 
            e_soft=1,
            hard=0,
            e_hard=0,
            e_num_optimizing_binder_pos=8,
            iters=100,
        ):
            self.do_iter(boltz_model, opt, verbose=verbose)

        with torch.no_grad():
            self.logits = 2 * self.logits
        
        print(f"stage 3: annealing down")
        for opt in Annealer(
            e_temp=0.01,
            hard=0,
            e_hard=0,
            num_optimizing_binder_pos=8,
            e_num_optimizing_binder_pos=12,
            iters=100,
        ):
            self.do_iter(boltz_model, opt, verbose=verbose)
        
        print(f"stage 4: argmax")
        for opt in Annealer(
            temp=0.01,
            e_temp=0.01,
            num_optimizing_binder_pos=12,
            e_num_optimizing_binder_pos=16,
            iters=10,
        ):
            self.do_iter(boltz_model, opt, verbose=verbose)
            

    def do_lmpnn_redesign(self, boltz_model, design_dir, structs, num_seqs=1):
        print("Running LigandMPNN redesign...")
        t0 = time.perf_counter()
        
        try:
            motif_indices = self.fixed_mask.nonzero(as_tuple=True)[0].tolist()
        except Exception:
            motif_indices = None

        lmpnn_seqs, fasta_path, best_sample_idx_by_state = perform_tied_lmpnn_redesign(
            design_dir=design_dir,
            state_results=structs,
            num_seqs=num_seqs,
            motif_indices=motif_indices,
        )

        regen_dir = os.path.join(design_dir, "lmpnn", "boltz_regen")
        os.makedirs(regen_dir, exist_ok=True)

        for seq_idx, seq in enumerate(lmpnn_seqs):
            if seq_idx == 0:
                continue  # skip original sequence
            print(f"Regenerating structures for LigandMPNN seq {seq_idx}")
            regen_structs = self.get_final_structs(boltz_model, samples=5, set_seq=seq)
            mydesign_utils.save_structs(regen_structs, regen_dir, prefix=f"lmpnn_seq{seq_idx}_")

        t1 = time.perf_counter()
        print(f"LigandMPNN redesign completed in {t1 - t0:.1f}s")

        return lmpnn_seqs, fasta_path, regen_dir


    def save_visualization_info(self, design_dir):
        save_path = os.path.join(design_dir, "visualization_info.pt")
        torch.save(
            {
                "pseudo_logit_traj": self.pseudo_logit_traj,
                "loss_log": self.loss_log,
                "final_seq": self.get_seq(),
                "num_iters": len(self.pseudo_logit_traj),
            },
            save_path,
        )

        print(f"Saved visualization information (logits, loss, etc.) to: {save_path}")