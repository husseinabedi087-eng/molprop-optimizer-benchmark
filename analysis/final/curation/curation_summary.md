# Data curation check: duplicates and cross-split overlap

Read-only check (2026-09-29) on the exact molecules and Bemis–Murcko scaffold split used in the study
(`results/xgb_mh/features/<dataset>.npz`: SMILES, labels, train/validation/test positions). RDKit 2026.03.6;
standard InChIKey. Script: `curation_check.py` (usage: `python curation_check.py <detail.csv>`);
per-pair detail: `curation_duplicates_detail.csv`. No split or result was changed.

Input files (sha256):

| file | sha256 |
|---|---|
| ESOL.npz | 01e70848719e0f8cab3b539b0426bae576eaad0894e888d42a1cb22c2969f66d |
| FreeSolv.npz | 31c5357f3476298fcefe6b8422878181aa6e9d9fd79a17236fa133147e75822d |
| Lipophilicity.npz | 9608473f13da700c3e98616897df8f75d01385326a549b27b2ba4c002c0fef61 |
| BBBP.npz | 4bcb42b077a6e2b1db19fad3ad036cc227c4ec9a170f0d9fa80d5a0916a7a92f |
| BACE.npz | fbc1eef51b854bee38e4917d75e23992f7f062afe062e58018754bef66002ed9 |

Definitions: *duplicate* = same full InChIKey; *extra copies* = molecules beyond the first in a duplicate group;
*variant* = same InChIKey first block (connectivity) but different full key (stereochemistry and/or
protonation). InChIKey generation failed for 0 molecules.

## ESOL (1,128 molecules)

| | train | val | test |
|---|---|---|---|
| duplicate groups (extra copies) | 9 (9) | 2 (2) | 0 |
| label spread within groups, max \|Δ logS\| | 1.03 | 0.10 | – |

Across splits: 0 shared (train–val, train–test, val–test); 0 variants.

## FreeSolv (642 molecules)

No duplicates within or across splits; 0 variants.

## Lipophilicity (4,200 molecules)

No duplicates within or across splits; 0 variants.

## BBBP (2,039 molecules)

| | train | val | test |
|---|---|---|---|
| duplicate groups (extra copies) | 62 (66) | 0 | 1 (1) |
| groups with conflicting class labels | 11 | – | 0 |

Across splits: **train–val: 1 exact duplicate, same class** (imidazole N–H tautomer written differently;
QKDDJDBFONZGBW-UHFFFAOYSA-N); train–test 0; val–test 0; 0 variants.

## BACE (1,513 molecules)

No duplicates within splits; 0 exact duplicates across splits. Variants across splits:

| split pair | groups | pairs | labels |
|---|---|---|---|
| train–val | 2 | 2 | 2/2 same class |
| train–test | 7 | 8 | 8/8 same class |
| val–test | 2 | 2 | 2/2 same class |

Of the 12 pairs: 2 stereo-only (one compound: unspecified vs specified stereocentres, and a second
diastereomer), 2 protonation-only (neutral vs protonated amidine/guanidine), 8 both.

## Why the scaffold split lets these through

Scaffolds were computed from the SMILES as written. A different tautomer or protonation form moves a double
bond or charge and changes the scaffold string, so copies of the same molecule can fall into different splits.
In all 13 cross-split cases above the scaffold used by the split differs between the two copies.

## Verdict

**1 exact cross-split duplicate in total (BBBP, train–val, labels agree), plus 11 protonation/stereo variant
groups (12 pairs) in BACE, 9 of them involving the test split, all with the same class; none in ESOL,
FreeSolv or Lipophilicity.** Within-split duplicates add label noise to training data (BBBP: 11 groups with
conflicting classes; ESOL: logS differences up to 1.03) but do not leak into test.
