# Denovoval RFD3 partial-diffusion chemistry note

## Scope and conclusion

This note records the 2026-07-22 inspection of the current denovoval RFD3
partial-diffusion staging and input-build path:

- source CIFs:
  `/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval/cifs`;
- staged/input artifacts:
  `/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval/ensembles`;
- RFD3 implementation:
  `/home/users/zhkim216/code/elix/foundry/models/rfd3`.

The current denovoval generation is not affected by a covalent-ligand bond-loss
bug because none of its 3,400 source CIFs explicitly encodes a protein-ligand
covalent bond. The bug is a latent limitation of the RFD3 dialect-2
partial-diffusion composition path: if an input does contain a cross-polymer /
non-polymer covalent bond, that bond is lost while the protein and ligand are
split and reassembled.

This is not evidence that any denovoval GLU or MET ligand is covalently linked.
The synthetic reproduction described below deliberately inserted a chemically
meaningless cross-chain bond only to test graph preservation.

## Current denovoval input audit

The preparation manifest reports:

- 3,400 source CIFs across 154 CCDs;
- 6,800 prepared inputs for `partial_t=2` and `partial_t=5`;
- zero missing inputs, preparation failures, duplicate sample IDs, or consumer
  validation failures.

The source-CIF audit found:

- binder chain `A` contains only the 20 standard amino-acid residue names;
- no modified protein residue is present on chain `A`;
- all context-chain `L` atoms are serialized as `HETATM`;
- `_struct_conn` occurs in zero source CIFs;
- the `covale` connection type occurs in zero source CIFs.

Thus, the current files represent the ligand as a separate non-polymer context,
without an explicit protein-ligand graph edge. This is the expected contract for
the current denovoval inputs.

## Amino-acid-named ligands

`GLU` and `MET` are the two standard-amino-acid CCD names used as ligands in the
current cohort. Each has 20 source conformers, or 80 prepared input rows after
the two partial-diffusion conditions are included.

At inspection time, the editable Foundry working tree classified AtomWorks
`NON_POLYMER` components using `chain_type` rather than the CCD name alone. The
production input path was checked through
`DesignInputSpecification.to_pipeline_input()`:

- `GLU`: 10 ligand atoms, all `is_ligand=True`, none `is_protein=True`;
- `MET`: 9 ligand atoms, all `is_ligand=True`, none `is_protein=True`.

The four focused non-polymer type-assignment regression tests also passed.
However, this behavior was present as an uncommitted modification to
`models/rfd3/src/rfd3/transforms/util_transforms.py`, with an untracked
`models/rfd3/tests/test_non_polymer_type_assignment.py`, at the time of the
inspection. A clean Foundry checkout will not contain that correction until it
is committed or otherwise preserved.

## Modified residues

Modified protein residues are not exercised by the current denovoval cohort.
A synthetic `AG(PTR)(SEP)SA` check was therefore used only to inspect the generic
path. In that check:

- `PTR` and `SEP` remained classified as protein rather than ligand;
- the `PTR`-`SEP` backbone bond was preserved through the partial-diffusion
  input build;
- the adjacent diffused-to-PTM backbone bonds were also reconstructable with
  the current component syntax.

One explicit semantic exception remains: RFD3's `inference_load_()` inherits
AtomWorks `STANDARD_PARSER_ARGS`, for which `convert_mse_to_met=True`. Therefore
an input `MSE` is normalized to `MET`; exact MSE CCD identity is not preserved.

## Covalently linked ligand examples

The Foundry regression-test source documents representative covalent cases:

- `4QDV`: protein `A:TYR143:OH` covalently linked to ligand
  `E:30U401:S1`;
- `8F7T`: protein `C:ASN403:ND2` covalently linked to glycan
  `G:NAG1:C1`;
- `1UA0`: DNA `DG4:C8` covalently linked to the `AF333:N` adduct;
- `6U6K`: cyclic thioether connections involving `ACE`, `TRP`, and `CYS`.

For the first example, the required graph is conceptually:

```text
protein A: ... TYR143-OH ---- S1-30U401 : ligand E
                              covalent
```

In an mmCIF input, this type of inter-component connection is normally supplied
through `_struct_conn` with a `covale` connection type.

## Confirmed partial-diffusion bond-loss mechanism

The RFD3 dialect-2 partial-diffusion build currently performs these operations:

1. It classifies the loaded input and subsets it to protein, DNA, and RNA atoms.
   This removes the non-polymer ligand from the working array.
2. It extracts the selected ligand independently from the original input.
3. It concatenates the polymer array and ligand array.

The coordinates and atoms survive, but a bond whose endpoints were separated
between the two slices cannot survive their independent slicing and plain
concatenation:

```text
source:  [protein atom] ---- [ligand atom]
                         bond

split:   [protein atom]      [ligand atom]

built:   [protein atom]      [ligand atom]
                         no bond
```

A mutation-free in-memory reproduction added one synthetic cross-chain bond to
the parsed `GLU_len150_0` AtomArray and ran the normal partial-diffusion build.
The observed counts were:

```text
source_cross_chain_bond_count = 1
built_cross_chain_bond_count  = 0
```

This establishes bond loss for that representation boundary. It does not claim
that the denovoval GLU input originally contained such a bond.

The downstream RFD3 transform
`FlagAndReassignCovalentModifications` detects polymer/non-polymer modifications
from the inter-PN-unit bond graph. If the bond has already been lost, the
transform cannot flag and atomize the attached polymer residue as a covalent
modification. Both partners may still have coordinates, but the model-visible
chemistry graph is then noncovalent.

There is also a narrower restoration issue in the general component-accumulation
path: `_restore_component_bonds()` is guarded by
`_check_has_backbone_connections_to_nonstandard_residues()`, which only detects
nonstandard backbone `C-N` bonds. A side-chain-to-ligand connection such as
`TYR-OH -- S1-30U` does not satisfy that guard.

## Validation gap and operational consequence

The denovoval staging validator currently checks source/staged digests, chain and
CCD identity, atom counts, ligand/polymer classification, and conditioning
masks. It does not compare source and built cross-residue bond graphs. A future
covalent input could therefore pass staging validation while losing its
protein-ligand bond during the consumer build.

The existing Foundry covalent-ligand regression tests depend on external CIF
fixtures under `models/rfd3/tests/test_data/` and skip when those files are
absent. That fixture directory was absent during this inspection, so the real
`4QDV`/`8F7T` tests did not protect this path.

Operationally:

- the current denovoval partial-diffusion jobs do not need to be stopped for
  this finding;
- the current cohort has no modified protein residues and no explicitly encoded
  covalent protein-ligand cases;
- before this path is used for a dataset containing covalent adducts,
  protein-ligand bonds must be restored after ligand append and validated against
  the source graph;
- a fixture-independent regression test should require the source cross-PN-unit
  bond set to equal the built bond set;
- whether `MSE -> MET` normalization is acceptable should be decided separately
  from covalent-bond preservation.

No RFD3 implementation fix was made as part of this diagnosis.
