import pickle
import numpy as np
import torch
import os
import glob
from utils import protein
from utils.geometry import compute_rmsd
from utils.residue_constants import atom_order

def motifRMSD(motif_pdb, design_pdb, motif_mask):
    with open(motif_pdb) as f:
        motif = protein.from_pdb_string(f.read())
    with open(design_pdb) as f:
        design = protein.from_pdb_string(f.read())

    true_motif_ca = torch.from_numpy(motif.atom_positions[:, atom_order["CA"]])
    design_motif_ca = torch.from_numpy(design.atom_positions[:, atom_order["CA"]][motif_mask])
    
    return compute_rmsd(design_motif_ca, true_motif_ca)

def compute_RG(design_pdb):
    with open(design_pdb) as f:
        design = protein.from_pdb_string(f.read())
    coords = design.atom_positions[:,1]
    return np.square(coords - coords.mean(0)).sum(-1).mean() ** 0.5
    
def caRMSD(pdb_path_a, pdb_path_b):
    with open(pdb_path_a) as f:
        prot_a = protein.from_pdb_string(f.read())
    with open(pdb_path_b) as f:
        prot_b = protein.from_pdb_string(f.read())

    ca_a = torch.from_numpy(prot_a.atom_positions[:, atom_order["CA"]])
    ca_b = torch.from_numpy(prot_b.atom_positions[:, atom_order["CA"]])
    n = min(len(ca_a), len(ca_b))
    ca_a, ca_b = ca_a[:n], ca_b[:n]
    
    return compute_rmsd(ca_a, ca_b)

def motif_results(motif_out_dir, motif_pdb, motif):
    
    jobs = sorted(glob.glob(os.path.join(motif_out_dir, "design*")))
        
    if args.workers > 1:
        from multiprocessing import Pool
        p = Pool(args.workers)
        p.__enter__()
        __map__ = p.imap
    else:
        __map__ = map
    rowss = list(tqdm.tqdm(__map__(do_single_dir, jobs), total=len(jobs)))
    if args.workers > 1:
        p.__exit__(None, None, None)
    df = []
    for rows in rowss:
        df.extend(rows)
        
    full = pd.DataFrame(df)
    
    agg = full.groupby(["design", "state"]).agg(
        motifRMSD_mean=("motifrmsd", "mean"),
        motifRMSD_std=("motifrmsd", "std"),
        plddt=("plddt", "mean"),
        ptm=("ptm", "mean"),
        iptm=("iptm", "mean"),
        radius_gyr=("radius_gyr", "mean"),
    ).reset_index()

    agg = agg.pivot(index="design", columns="state").reset_index()
    agg.columns = ["_".join(map(str, col)).rstrip("_") for col in agg.columns.to_flat_index()]
    agg = agg.rename(columns=lambda c: c.replace("_0", "_unbound").replace("_1", "_bound"))
    
    agg.to_csv(os.path.join(motif_out_dir,"_aggresults.csv"),index=False)
    full.to_csv(os.path.join(motif_out_dir,"_fullresults.csv"),index=False)



        
def do_single_dir(design_dir):
    rows = []
    design_name = os.path.basename(design_dir)
    with open(os.path.join(design_dir, f"{motif}_spec.pkl"), "rb") as f:
        motif_mask = pickle.load(f)["motif_mask"]

    for state in [0, 1]:
        
        if args.lmpnn:
            pkl_path = f"{design_dir}/lmpnn/boltz_regen/lmpnn_seq1_state{state}.pkl"
        else:
            pkl_path = f"{design_dir}/state{state}.pkl"
        with open(pkl_path, 'rb') as f:
            outdict = CPU_Unpickler(f).load()

        for sample_idx in range(5):  # assume 5 samples per state
            for seqid in range(1, 9): # num lpnn seqs
                if args.lmpnn:
                    pdb_file = f"{design_dir}/lmpnn/boltz_regen/lmpnn_seq{seqid}_state{state}_sample{sample_idx}.pdb"
                else:
                    pdb_file = f"{design_dir}/state{state}_sample{sample_idx}.pdb"
                
                if not os.path.exists(pdb_file):
                    continue
            
                rmsd = motifRMSD(motif_pdb, pdb_file, motif_mask)
                RG = compute_RG(pdb_file)
                rows.append({
                    "design": f"{design_name}:{seqid}" if args.lmpnn else design_name,
                    "state": state,
                    "sample": sample_idx,
                    "motifrmsd": rmsd.item() if hasattr(rmsd, "item") else float(rmsd),
                    "radius_gyr": RG,
                    "plddt": outdict["plddt"].cpu().numpy()[sample_idx].mean(),
                    "ptm": outdict["ptm"].cpu().numpy()[sample_idx],
                    'iptm': outdict["ligand_iptm"].cpu().numpy()[sample_idx],
                })
                if not args.lmpnn: break
                    
    return rows
    
if __name__ == "__main__":
    import io
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            else: return super().find_class(module, name)
    
    
    
    import sys, tqdm
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--lmpnn', action='store_true')
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args()
    
    import torch
    import pandas as pd

    
    out_dir = args.dir
    for motif in os.listdir(out_dir):
        print(motif)
        motif_pdb = f"motifs/{motif}.pdb"
        motif_out_dir = os.path.join(out_dir,motif)
        motif_results(motif_out_dir, motif_pdb, motif)
