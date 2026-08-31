# denovoval LASErMPNN chemistry and reconstruction contract

## Authoritative inputs

- Sample membership and coordinates come only from
  `/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval/sampling_inputs.csv`
  and its referenced canonical CIFs.
- CCD chemistry comes from
  `/scratch/users/zhkim216/datasets/ccd_mirror/components.cif`. The per-CCD files
  under the same mirror are its indexed representation and are used for
  efficient component lookup.
- Small-molecule protonation uses only the `SMILES_CANONICAL` descriptor whose
  program is `OpenEye OEToolkits`. The descriptor's `program_version` is
  component metadata recorded by wwPDB, so it may differ between CCD entries.

## PDB transport and protonation

1. Canonical CIFs are converted to staging PDBs. CCD codes longer than three
   characters and amino-acid CCD codes receive deterministic, collision-free
   PDB aliases. CCD heavy-atom `CONECT` records are written without
   distance-based bond inference.
2. For 133 small-molecule CCDs, the repository calls the unmodified
   `/home/users/zhkim216/code/NISE/protonate_and_add_conect_records.py` entrypoint
   with the OpenEye canonical SMILES.
3. Five diagnosed stock-NISE failures (`A1L3W`, `A1LYJ`, `CLA`, `HEM`, and
   `MF8`) use `atomworks.io.tools.rdkit.ccd_code_to_rdkit(...,
   hydrogen_policy="infer")`. Their PDB metadata is normalized to the same
   transport convention as stock NISE. This is an explicit allowlist, not a
   generic fallback.
4. A full 3,400-conformer pass found 19 additional, conformer-specific stock
   failures. Stock NISE's ProDy ligand rewrite drops the staging `CONECT`
   records, so RDKit reconstructs a proximity graph; for example,
   `2TA_len300_1` receives 41 inferred bonds although its CCD has 40, while a
   passing 2TA conformer has 40/40. These 19 pinned `sample_id` values use the
   same AtomWorks route. Other conformers of those CCDs remain on stock NISE.
   Any unconfigured stock failure stops preparation.
5. The 16 metal CCD families (640 samples) are copied through without
   protonation.

Stock NISE intentionally converts the ligand to transport chain `B`, residue
number `1`, and sequential element-based atom names, while setting occupancy to
`1.0` and B-factor to `0.0`. The pipeline preserves that output for LASErMPNN.
It does not restore canonical heavy metadata or formal charges before sampling.

## Atom-mapping sidecar

Each prepared PDB has a CSV sidecar in `staging/atom_mappings/`. It records:

- transport atom index, serial, name, element, chain, residue ID, and alias;
- the canonical source identity for each heavy atom;
- the heavy parent of every generated hydrogen.

Sampling manifests carry both the sidecar path and SHA-256 digest. Backmapping
requires those values to agree with the staging manifest, first proves that the
sampled transport ligand is byte-semantically unchanged in names, elements,
order, and coordinates, and then uses the sidecar to restore canonical heavy
atom names, chain `L`, source residue ID, and the original CCD code. Generated
hydrogens are retained and receive collision-free names (`H001`, `H002`, ...).

## Resolved failure modes

- CCD-wide and conformer-specific stock-NISE failures were reproduced rather than inferred from the
  bridge. They arise from PDB proximity-derived graphs or explicit-stereo-H and
  metal-coordination incompatibilities with RDKit template assignment.
- The previously observed H-parent mismatches for 27 CCDs are not an error
  under the stock-NISE/OpenEye contract. They do not trigger fallback.
- Formal charge is not restored from the source or CCD after protonation. The
  transport PDB retains the charge assignment produced by the selected
  protonation route.
- Amino-acid-named ligands are protected from NISE's protein-track heuristic by
  their PDB alias and, if necessary, an in-memory-only atom-name mask. The
  serialized ligand remains immutable, and synthetic `CAP` residues are
  forbidden.
- ATOM/HETATM serials must be exactly `1..N`; every `CONECT` endpoint must
  reference an existing atom.

## Sampling contract

- weights: `laser_weights_0p1A_nothing_heldout.pt` with pinned SHA-256;
- sequence, first-shell sequence, and chi temperatures: `1e-6`;
- disabled residues: `X`;
- sequence/chi min-p: `0.0`;
- Ala/Gly budgets: stock defaults `4/0` (the optional constraint flag is not
  enabled);
- two sequences per input for the full production run.

## Validation status

The implementation must pass focused unit tests, all-CCD preparation probes,
representative SAM/metal/explicit-exception sampling, sidecar backmapping, AF3
JSON inspection, and self-consistency metrics before production artifacts are
accepted. Runtime evidence is recorded separately under the scratch output
root after clean regeneration; prior results from the superseded chemistry
contract are not evidence for this version.
