import pickle
import numpy as np
import torch
import os
import glob
from utils import protein
from utils.geometry import compute_rmsd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str)
parser.add_argument('--workers', type=int, default=0)
args = parser.parse_args()

def motifRMSD(design_pdb):
    with open(motif_pdb) as f:
        motif = protein.from_pdb_string(f.read())
    with open(design_pdb) as f:
        design = protein.from_pdb_string(f.read())

    true_motif_ca = torch.from_numpy(motif.atom_positions[:, 1])
    design_motif_ca = torch.from_numpy(design.atom_positions[:, 1][motif_mask])
    
    return compute_rmsd(design_motif_ca, true_motif_ca)

def get_rmsd(prot1, prot2):
    return compute_rmsd(
        torch.from_numpy(prot1.atom_positions[:50,1]),
        torch.from_numpy(prot2.atom_positions[:50,1]),
    )


def do_job(design_dir):
    
    state0 = []
    state1 = []
    state2 = []
    with open(os.path.join(design_dir, f"state0.pkl"), 'rb') as f:
        outdict0 = CPU_Unpickler(f).load()
    with open(os.path.join(design_dir, f"state1.pkl"), 'rb') as f:
        outdict1 = CPU_Unpickler(f).load()
    with open(os.path.join(design_dir, f"state2.pkl"), 'rb') as f:
        outdict2 = CPU_Unpickler(f).load()
    for i in range(5):  # assume 5 samples per state
        with open(f"{design_dir}/state0_sample{i}.pdb") as f:
            state0.append(protein.from_pdb_string(f.read()))
        with open(f"{design_dir}/state1_sample{i}.pdb") as f:
            state1.append(protein.from_pdb_string(f.read()))
        with open(f"{design_dir}/state2_sample{i}.pdb") as f:
            state2.append(protein.from_pdb_string(f.read()))
    
    cross_rmsd01 = [get_rmsd(state0[i], state1[j]).item() for i in range(5) for j in range(5)]
    cross_rmsd12 = [get_rmsd(state1[i], state2[j]).item() for i in range(5) for j in range(5)]
    cross_rmsd02 = [get_rmsd(state0[i], state2[j]).item() for i in range(5) for j in range(5)]
    
    intra_rmsd0 = [get_rmsd(state0[i], state0[j]).item() for i in range(5) for j in range(i+1, 5)]
    intra_rmsd1 = [get_rmsd(state1[i], state1[j]).item() for i in range(5) for j in range(i+1, 5)]
    intra_rmsd2 = [get_rmsd(state2[i], state2[j]).item() for i in range(5) for j in range(i+1, 5)]
    
    # breakpoint()
    row = {
        "sample": os.path.basename(design_dir ),
        "plddt_0": outdict0["plddt"].cpu().numpy().mean(),
        "ptm_0": outdict0["ptm"].cpu().numpy().mean(),
        "plddt_1": outdict1["plddt"].cpu().numpy().mean(),
        "ptm_1": outdict1["ptm"].cpu().numpy().mean(),
        "plddt_2": outdict2["plddt"].cpu().numpy().mean(),
        "ptm_2": outdict2["ptm"].cpu().numpy().mean(),
        'cross_rmsd01_mean': np.mean(cross_rmsd01),
        'cross_rmsd01_std': np.std(cross_rmsd01),
        'cross_rmsd02_mean': np.mean(cross_rmsd02),
        'cross_rmsd02_std': np.std(cross_rmsd02),
        'cross_rmsd12_mean': np.mean(cross_rmsd12),
        'cross_rmsd12_std': np.std(cross_rmsd12),
        'intra_rmsd0_mean': np.mean(intra_rmsd0),
        'intra_rmsd0_std': np.std(intra_rmsd0),
        'intra_rmsd1_mean': np.mean(intra_rmsd1),
        'intra_rmsd1_std': np.std(intra_rmsd1),
        'intra_rmsd2_mean': np.mean(intra_rmsd2),
        'intra_rmsd2_std': np.std(intra_rmsd2),
    }
    return row
    
if __name__ == "__main__":
    import io
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            else: return super().find_class(module, name)
    
    
    
    import torch
    import pandas as pd
    import sys, tqdm
    

    rows = []

    jobs = []
    for design_dir in tqdm.tqdm(sorted(glob.glob(os.path.join(args.dir, "design*")))):
        jobs.append(os.path.join(args.dir, os.path.basename(design_dir)))

    from multiprocessing import Pool
    if args.workers > 1:
        p = Pool(args.workers)
        p.__enter__()
        __map__ = p.imap
    else:
        __map__ = map
    rows = list(tqdm.tqdm(__map__(do_job, jobs), total=len(jobs)))
    if args.workers > 1:
        p.__exit__(None, None, None)

    pd.DataFrame(rows).to_csv(f"{args.dir}/results.csv", index=False)


# def main():
#     dirs = os.listdir(args.mmcif_dir)
#     files = [os.listdir(f"{args.mmcif_dir}/{dir}") for dir in dirs]
#     files = sum(files, [])
#     info = []
#     for inf in infos:
#         info.extend(inf)
#     df = pd.DataFrame(info).set_index("name")
#     df.to_csv(args.outcsv)

    # agg = full.groupby(["design", "state"]).agg(
    #     motifRMSD_mean=("motifrmsd", "mean"),
    #     motifRMSD_std=("motifrmsd", "std"),
    #     plddt=("plddt", "mean"),
    #     ptm=("ptm", "mean")
    # ).reset_index()

    # agg = agg.pivot(index="design", columns="state").reset_index()
    # agg.columns = ["_".join(map(str, col)).rstrip("_") for col in agg.columns.to_flat_index()]
    # agg = agg.rename(columns=lambda c: c.replace("_0", "_unbound").replace("_1", "_bound"))
    
    # agg.to_csv(os.path.join(motif_out_dir,"_aggresults.csv"),index=False)
    # full.to_csv(os.path.join(motif_out_dir,"_fullresults.csv"),index=False)


        
    # # motif = "3ixt"
    # # design_dir = f"./out/onemotif_twostates/{motif}/design1/"

    # # with open(os.path.join(design_dir, f"{motif}_spec.pkl"), "rb") as f:
    # #     motif_mask = pickle.load(f)["motif_mask"]

    # motif_pdb = f"/data/cb/mihirb14/projects/BoltzDesign1/motifs/{motif}.pdb"

    # pdb_files = sorted(glob.glob(os.path.join(design_dir, "state*_sample*.pdb")))

    # from collections import defaultdict
    # state_groups = defaultdict(list)
    # for pdb in pdb_files:
    #     base = os.path.basename(pdb)
    #     state = base.split("_")[0]
    #     state_groups[state].append(pdb)

    # for state, files in state_groups.items():
    #     print(f"\n{state}:")
    #     for fpath in sorted(files):
    #         rmsd = motifRMSD(motif_pdb, fpath, motif_mask)
    #         print(f"  {os.path.basename(fpath)} → RMSD {rmsd:.3f}")
