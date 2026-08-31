# Denovoval LASErMPNN validation contract

This checklist is the stop/go contract for the canonical 3,400-input denovoval
run. Full AF3 must not be submitted unless the full sampling and backmapping
gates are complete with zero errors.

## Pinned experiment

- Source rows: `sampling_inputs.csv`, exactly 3,400 unique sample IDs.
- Coverage: 1,700 length-150 and 1,700 length-300 inputs; small molecules and
  metals; 154 unique CCD codes.
- LASErMPNN: `laser_weights_0p1A_nothing_heldout.pt`, two designs per input,
  sequence/first-shell/chi temperature `1e-6`, sequence/chi min-p `0.0`, and
  disabled residues `X`.
- Full layout: 20 shards, 340 design rows per shard, 6,800 sampled PDBs.
- AF3 single-sequence mode: seed 42, no MSA, no templates, 10 recycles, and five
  diffusion samples per design (34,000 predictions).

## Chemistry-input interpretation

- LASErMPNN consumes the protonated PDB, not a SMILES string. Its upstream
  requirement is that the ligand carry the appropriate hydrogens for the
  intended protonation state.
- Neither the NISE nor LASErMPNN upstream interface requires OpenEye, canonical,
  or isomeric SMILES specifically. The stock NISE helper parses the supplied
  string with RDKit, transfers its bond orders to the PDB heavy-atom graph, and
  adds hydrogens; it does not select a pH-dependent protonation state or
  enumerate tautomers.
- This experiment therefore pins the OpenEye `SMILES_CANONICAL` descriptor in
  the CCD mirror as a deterministic source policy, not as an upstream
  LASErMPNN requirement. The descriptor must still parse, map to the staged
  heavy-atom graph, and represent the ligand state intended for this benchmark.
- Staging gates prove template/mapping compatibility, atom and coordinate
  integrity, and reproducibility. They do not prove that the CCD descriptor is
  the biologically correct protonation state at a particular pH. A deliberately
  different protonation state or tautomer must be introduced as a new pinned
  chemistry input and digest, never as a silent substitution.

## Sampling gate

Run:

```bash
python -m allatom_design.eval.benchmarks.denovoval.lasermpnn.validate_outputs \
  sampling --num-shards 20
```

The gate checks all of the following, not only aggregate counts:

- exactly `shard_00.csv` through `shard_19.csv`, the expected modulo-shard
  membership, and the exact 3,400 x `{1,2}` design-key matrix;
- exactly 6,800 PDB files, with no stale or extra PDBs;
- sampled-PDB, protonated-input, atom-mapping, and model-weight SHA256 values;
- pinned temperatures, min-p values, disabled residues, ALA/GLY budgets, and
  deterministic per-input seed;
- finite LASEr NLL and binding-site NLL;
- canonical protein alphabet, exact requested sequence length, contiguous atom
  serials from 1, finite coordinates, and complete N/CA/C/O identity;
- stock-NISE backbone idealization diagnostics: input-to-output CA RMSD and
  N/CA/C displacement remain inside the validated frame envelope, psi-dependent
  O reconstruction is reported, and designs 1/2 have identical backbones;
- immutable ligand atom identity, order, chain, residue identity, element,
  coordinates, and atom count relative to the protonated input;
- the expected ligand in the LASEr ligand track, no synthetic `CAP`, and both
  small-molecule and metal coverage;
- a complete local decoding-order permutation for every design, a matching
  SHA256, and distinct decoding orders for the two designs of each input.

The authoritative report is `sampling/validation.json`. Any error means stop:
do not backmap or submit AF3; report the affected design IDs and diagnostics.

## Backmapping gate

Run `backmap_designs.py` only from the manifests that passed the sampling gate.
Its `validation.json` must prove:

- exactly 6,800 backmapped CIFs and the complete design-key matrix;
- `sampling_manifest_path` points to the current shard for every design;
- the sampled PDB digest and atom-mapping sidecar digest still match;
- canonical protein chain `A`, ligand chain `L`, and original CCD code are
  restored, including PDB-safe aliases and amino-acid ligands;
- ligand coordinates are unchanged, protein sequence is unchanged, entity
  categories are valid, and categorical mismatch count is zero.

## AF3 input gate

First run every AF3 shard with `ACTION=generate-inputs-only`, then run:

```bash
python -m allatom_design.eval.benchmarks.denovoval.lasermpnn.validate_outputs \
  af3-inputs --num-arrays 20
```

The gate checks exactly 20 status CSV/JSON pairs, current design-manifest SHA256
and chunk slices, exactly 6,800 JSON files, all 154 CCD codes, chain `A` protein
sequence, chain `L` ligand CCD, seed 42, AF3 dialect/version, and blank paired
and unpaired MSA plus no templates. Full AF3 is allowed only when
`reports/validation_inputs.json` is complete with zero errors.

## AF3 prediction and metric gate

After all full AF3 shards are terminal and retries are reconciled, run:

```bash
python -m allatom_design.eval.benchmarks.denovoval.lasermpnn.validate_outputs \
  af3-complete --num-arrays 20 --predictions-per-design 5
```

The gate checks:

- exactly 6,800 complete status rows and prediction directories;
- five valid predictions per design, no malformed/surplus prediction, current
  input fingerprint, and no metric error;
- exactly 34,000 self-consistency and 34,000 docking rows with the exact
  design/diffusion-index matrix;
- `metric_status=ok`, empty errors, and finite CA-RMSD/confidence/docking output;
- finite TM-score/TM-align fields, score in `[0,1]`, at least three matched CA
  atoms, and valid sample/prediction coverage.

The authoritative final report is `af3_ss/reports/validation_complete.json`.
Scheduler success alone is not completion; scheduler, artifact, metric, and
validation-report states must all agree.

## Smoke and retry checks

- Smoke uses SAM and MG, two designs each, two shards, one recycle, and one
  diffusion sample. Refresh backmapping from both current smoke manifests before
  regenerating AF3 reports.
- Smoke TM-score is required before full AF3 submission.
- Retry only failed/incomplete original shard IDs. Set
  `ALLOW_PARTIAL_ARRAY=true` for a subset array and keep `NUM_SHARDS=20` or
  `NUM_ARRAYS=20` so original shard identity is preserved.
- A command-line `sbatch --export=...` replaces the script's `#SBATCH --export`
  directive. If an explicit export list is used, retain
  `ELIX_REQUEUE_ON_USR2=1`; otherwise the pre-timeout signal terminates the task
  instead of requeueing it. The container wrapper also preserves this opt-in
  from the original sbatch directive when the variable is not explicitly
  overridden.
- Existing AF3 predictions may be reused only when strict input fingerprinting
  passes; status and metric reports must still reference the current design
  manifest SHA256.
