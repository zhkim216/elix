# SwitchCraft State-Switching Generation Plan

## Goal

Generate 26,023 SwitchCraft design trajectories for the state-switching
denovoval benchmark and store the results under:

```text
/scratch/users/zhkim216/experiment_result_analysis/benchmarks/state_switching/denovoval
```

The requested totals are:

| Design family | Number of trajectories |
| --- | ---: |
| Positive allostery: 24 motifs x 5 ligands x 100 | 12,000 |
| Negative allostery: 24 motifs x 5 ligands x 100 | 12,000 |
| Motif switching | 1,000 |
| Ligand modification | 558 |
| Ligand discrimination | 465 |
| **Total** | **26,023** |

## Sources of Truth

The scientific task definitions come from the local SwitchCraft paper:

```text
/home/users/zhkim216/code/switchcraft/2026_switchcraft.pdf
```

The paper defines the experiment families and the roles of the states, motifs,
and ligands. It does not enumerate the chain and residue range for every one of
the 24 motif-scaffolding problems. Exact motif extraction and scaffold-length
specifications therefore come from the `REMARK 999` records in:

```text
/home/users/zhkim216/code/switchcraft/motifs/*.pdb
```

The runtime consumer is:

```text
/home/users/zhkim216/code/switchcraft/switchcraft.py
```

Run-specific YAML files are generated and frozen under
`denovoval/manifests/configs/`. The upstream `tasks/*.yaml` files are not
modified because several of them are examples or placeholders rather than
complete reproductions of the paper experiments.

## RFDiffusion Motif Benchmark Set

Positive and negative allostery will each cover the 24 RFDiffusion
motif-scaffolding problems used by the paper:

```text
1bcf
1prw
1qjg
1ycr
2kl8
3ixt
4jhw
4zyp
5ius
5tpn
5trv_long
5trv_med
5trv_short
5wn9
5yui
6e6r_long
6e6r_med
6e6r_short
6exz_long
6exz_med
6exz_short
7mrx_128
7mrx_85
7mrx_60
```

The three local `7mrx` specifications correspond to the benchmark's long,
medium, and short variants, respectively. The additional local motif files
`1anf`, `1cfd`, and `1cll` are not part of this 24-problem matrix.

SwitchCraft does not run RFDiffusion or Genie2 during generation. It uses the
benchmark motif coordinates and the Genie2-style motif specifications already
encoded in the local PDB files, then samples the allowed scaffold-segment
lengths in `utils/motif_utils.py` for every trajectory.

## Allostery Matrix

Each allostery direction will cover the Cartesian product of 24 motifs and five
ligands, for 120 combinations per direction and 240 allostery cases in total:

| Paper ligand | SwitchCraft state entry |
| --- | --- |
| OQO | `ccd:OQO` |
| Flavin adenine dinucleotide | `ccd:FAD` |
| Zinc ion | `ccd:ZN` |
| Magnesium ion | `ccd:MG` |
| Double-stranded GAATTC DNA | `dna:GAATTC` |

For DNA, the runtime automatically adds the reverse-complement strand.

Positive allostery uses an inactive motif in the apo state and an active motif
in the ligand-bound state. Negative allostery reverses those motif roles. Both
states retain the default per-state `ContactLoss`, and the bound state retains
`LigandContactLoss`.

Each motif-ligand combination is a separate case and receives 100 independent
optimization trajectories in each allostery direction. This produces 12,000
positive-allostery trajectories and 12,000 negative-allostery trajectories.
The allocation follows the paper's `problem x specification x ligand` unit of
generation; the 100 trajectories are not divided across the matrix.

## Motif Switching

Motif switching will generate 1,000 trajectories with OQO as the effector:

```yaml
motifs:
  - 3ixt
  - 1ycr

states:
  - []
  - ["ccd:OQO"]
```

The state-specific motif roles are:

| State | 3IXT | 1YCR |
| --- | --- | --- |
| A: apo | active | inactive |
| B: OQO-bound | inactive | active |

The exact motif inputs are already encoded locally:

- `3ixt.pdb`: chain P, residues 254-277; N- and C-terminal scaffold
  segments each range from 10 to 40 residues.
- `1ycr.pdb`: chain B, residues 19-27; N- and C-terminal scaffold
  segments each range from 10 to 40 residues.

The two Genie2-style specifications are merged by `utils/motif_utils.py`. The
resulting layout is:

```text
scaffold(10-40)
-> 3IXT motif(24 residues)
-> scaffold(10-40)
-> 1YCR motif(9 residues)
-> scaffold(10-40)
```

The merged scaffold therefore has a sampled total length of 63-153 residues.
The motif residues and coordinates are fixed while the remaining sequence is
optimized.

## Ligand Modification

Ligand modification will generate 558 trajectories for a 50-residue protein
with heme in state A and oxygenated heme in state B. The executable
reconstruction of the paper specification is:

```yaml
length: 50

states:
  - ["ccd:HEM"]
  - ["ccd:HEM", "ccd:OXY"]

losses:
  - type: LigandContactLoss
    state: 0
    idx: 1
  - type: LigandContactLoss
    state: 1
    idx: 1
  - type: LigandContactLoss
    state: 1
    idx: 2
  - type: ConfChangeLoss
    state: [0, 1]
    strength: 10
```

The paper does not publish its literal YAML or CCD tokens. `HEM` plus molecular
oxygen `OXY` is the agreed code-grounded reconstruction of the paper's ligand A
to ligand A-plus-B formulation.

## Ligand Discrimination

Ligand discrimination will generate 465 trajectories for the paper's
50-residue, three-state system:

```yaml
length: 50

states:
  - []
  - ["ccd:OQO"]
  - ["ccd:CA"]
```

The OQO- and calcium-bound states each receive `LigandContactLoss`. All three
state pairs receive the default `ConfChangeLoss` setting from the task example,
with strength 10.

## Planned Runtime Shape

The existing CLI will remain the generation entrypoint. A thin Slurm array
launcher will select a task/config, output partition, `num_workers`, and
`worker_id`, then invoke:

```text
python switchcraft.py \
  --config <run-config.yaml> \
  --num_designs <task-or-combination-total> \
  --num_workers <array-worker-count> \
  --worker_id <zero-based-worker-id> \
  --outpath <task-output-directory>
```

The scripts will use the existing isolated SwitchCraft environment and the
container already validated by the interactive-node smoke run. Full generation
will use the normal four-stage optimization schedule and the default five final
Boltz-1 diffusion samples for every actual state. No `--debug`, `--length`,
`--recycles`, or `--ligandmpnn_seqs` override will be enabled. Thus all settings
other than the requested trajectory counts come from the frozen paper-derived
task definitions and SwitchCraft runtime defaults.

A full-setting smoke run was completed for positive allostery with the
`7mrx_128` motif and double-stranded `GAATTC` DNA on an A100-SXM4-80GB GPU. It
completed in 8 minutes 39 seconds wall-clock time, including 444.8 seconds for
optimization and 21.6 seconds for final structure generation. Peak observed GPU
memory was 7,166 MiB and host MaxRSS was 6,206,292 KiB. The production work
plan assigns at most 20 designs to each two-state worker and 15 designs to each
three-state ligand-discrimination worker. Every array task requests one A100,
H100, or H200 GPU for four hours. The smaller three-state chunk avoids
extrapolating the two-state smoke timing beyond the walltime.
The 1,309 work items are submitted as two scheduler arrays because Sherlock
limits an array to 1,000 tasks. Global work indices 0-999 are submitted to
`owners` without an array throttle. Global work indices 1000-1308 are submitted
independently to `possu` with `%8`, limiting that array to eight concurrent GPU
tasks. Both arrays accept only A100, H100, and H200 GPU SKUs.

With five diffusion samples per state, the final design tree is projected to
contain 603,621 files and 26,023 design directories, or 629,644 inodes
before case/config/manifest entries. The 1,309 array tasks add at most 2,618
stdout/stderr log files. At preparation time the user scratch quota had
13,064,930 of 50,000,000 inodes in use, so this run is expected to leave roughly
36.3 million user inodes free. Storage is inferred from the retained 3.6 MiB
two-state smoke output rather than fully benchmarked and is expected to be about
0.09-0.13 TiB.

## Output Layout and Run Identity

The requested scratch directory is the output root. Outputs will be partitioned
by scientific task and, for allostery, by ligand and motif. Every allostery
leaf contains 100 design indices. The exact hierarchy will follow the
existing behavior in which `switchcraft.py` appends the motif stem to the
supplied output path.

```text
denovoval/
  positive_allostery/<ligand>/<motif>/design<N>/
  negative_allostery/<ligand>/<motif>/design<N>/
  motif_switching/3ixt_1ycr/design<N>/
  ligand_modification/design<N>/
  ligand_discrimination/design<N>/
  manifests/
  logs/
```

The run manifest will record the SwitchCraft git commit, config digests,
environment and container paths, requested counts, matrix allocation, command
shape, and expected output directories. After generation begins, the staged
run configs and manifest become the source of truth for validation; validation
will not silently refresh task definitions from upstream files.

## Completeness Validation

Validation will operate on final design directories rather than Slurm state
alone. It will report, per task and allostery combination:

- requested, present, complete, and incomplete design counts;
- missing, duplicate, and unexpected design indices;
- missing state pickle files;
- missing PDB or CIF files among the five expected samples per state;
- missing motif specification pickle files for motif-containing tasks;
- nonzero extra, unresolved, or failed records recorded by the launcher.

A design is complete only when all state pickle files and all five PDB/CIF
sample pairs for every expected state exist. Motif-containing designs must also
contain the expected motif specification pickle files.

## Execution Authorization

Implementation, config/manifest preparation, dry-run submission validation,
and one interactive full-setting smoke were authorized. The 26,023-design
production array is submitted only through the explicit
`submit_generation.sh --submit` invocation; the helper remains dry-run-only by
default.

The production arrays were submitted on 2026-07-20 as job `34757708`
(`owners`, global work 0-999, unthrottled) and job `34757709` (`possu`, global
work 1000-1308, `%8`). The submission receipt is stored next to the frozen
manifest as `submission_20260720T095027Z.json`.

Those initial arrays were cancelled after compute nodes reported that the bare
`apptainer` command was absent from `PATH`. The launcher now uses the verified
absolute runtime `/bin/singularity`. Replacement jobs `34759028` (`owners`) and
`34759029` (`possu`) were submitted with the same work ranges and resource
policy; details are in `resubmission_20260720T100958Z.json`.

## Implemented Commands

Replace the superseded generation plan with the 243-case, 1,309-item work
plan:

```bash
./prepare_generation.py --replace-generation-plan
```

Print the resolved `sbatch` command without submitting it:

```bash
./submit_generation.sh
```

Submit only after a separate explicit authorization:

```bash
./submit_generation.sh --submit
```

Audit all production outputs against the frozen manifest:

```bash
./validate_generation.py
```

The validator exits with status 0 only when every requested design is complete.
It writes its JSON report to `denovoval/manifests/completeness_report.json`.
The array worker invokes `run_worker.py`, which skips complete design
directories and reruns only incomplete assigned indices through the existing
`switchcraft.py` entrypoint.

Smoke timing, GPU-memory polling, stdout, and validation artifacts are stored
under `denovoval/manifests/smoke/`. The completed smoke design is retained at:

```text
denovoval/positive_allostery/dna_GAATTC/7mrx_128/design0
```

## Done When

The implementation phase is complete when run-local configs, a thin Slurm
launcher, a dry-run-by-default submission helper, and a completeness validator
exist; syntax/config and matrix-count checks pass; and the authorized smoke
produces all expected artifacts without submitting the production array.
