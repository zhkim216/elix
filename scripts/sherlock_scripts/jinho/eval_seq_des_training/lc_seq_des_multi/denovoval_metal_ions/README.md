# denovoval_metal_ions lc_seq_des_multi on Sherlock

This handoff expects a flat CIF directory on Sherlock:

```text
/scratch/users/zhkim216/datasets/val_cifs/denovoval_cifs
```

The selected `.cif` files, sample list, and sampling CSV should live directly in that directory, not in a nested `cifs/` or `single_metal_ions/` subfolder:

```text
/scratch/users/zhkim216/datasets/val_cifs/denovoval_cifs/*.cif
/scratch/users/zhkim216/datasets/val_cifs/denovoval_cifs/denovoval_metal_ions.txt
/scratch/users/zhkim216/datasets/val_cifs/denovoval_cifs/sampling_inputs_denovoval_metal_ions.csv
```

The sweep filters compact CIF names such as `CA_len150_0.cif` by:

```text
pdb_cfg.sample_indices=[0,1,2,3,4]
pdb_cfg.sample_lengths=[150,250]
```

That selects 100 input CIFs per model/step before sequence sampling.

Submit the model/step sweep:

```bash
PROJECT_ROOT=/home/users/zhkim216/code/elix \
  bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container_elix.sh \
  scripts/sherlock_scripts/jinho/eval_seq_des_training/lc_seq_des_multi/denovoval_metal_ions/lc_seq_des_multi_denovoval_metal_ions_per_step_sweep.sbatch
```

Filter to one model and step:

```bash
PROJECT_ROOT=/home/users/zhkim216/code/elix MODEL_FILTER=proto_nprot1_external_evidence STEP_FILTER=42500 \
  bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container_elix.sh \
  scripts/sherlock_scripts/jinho/eval_seq_des_training/lc_seq_des_multi/denovoval_metal_ions/lc_seq_des_multi_denovoval_metal_ions_per_step_sweep.sbatch
```
