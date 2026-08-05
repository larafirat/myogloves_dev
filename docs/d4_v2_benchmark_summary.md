# D4_v2 Benchmark Summary

## What "Support Removal" Means

These benchmark scripts do **not** mean "the object falls by accident." They
mean we deliberately remove the pedestal's support after the hand has already
established contact, to test whether the grasp can actually carry load.

- `Retract baseline`:
  After `250` consecutive contact steps, the start pillar is moved sideways by
  `1.0 m`. This asks: does the hand keep the object once the table/pedestal is
  no longer underneath it?

- `Lift baseline`:
  After `250` consecutive contact steps, the start pillar is lowered by
  `0.20 m`. This asks: does the hand keep the object once vertical support is
  removed, which is a closer proxy to a grab-and-lift phase in our fixed-arm
  setup?

## Caption

**Caption:** Current comparison of `bare_msk`, restored-best working
`D4_v2_hybrid_per_finger_calibrated`, and the separate
`D4_v2_paper_like_thumb_calibrated` branch. Same object mass, same widened
pedestal, same grasp scene. `Contact (s)` is first hand-object contact time.
`Peak touch (s)` is the longest continuous hand-object contact streak achieved
before support removal. `Retract/Lift (s)` is when support removal happens.
`Held?` indicates whether the object remains effectively carried after support
is removed. `Touch digits` records which digits participated before support
removal. These tables are intended as a reference point for future controller
tuning, especially on the working `D4_v2` branch.

## Retract Baseline

| Object | Condition | Contact (s) | Peak touch (s) | Retract (s) | Held? | Drift (m) | Touch digits |
|---|---|---:|---:|---:|---|---:|---|
| `tall_box` | `bare_msk` | 1.120 | 0.520 | 1.618 | False | 1.371 | thumb |
| `tall_box` | `D4_v2_hybrid_per_finger_calibrated` | 1.234 | 0.572 | 1.732 | False | 1.370 | index,middle,ring,thumb |
| `tall_box` | `D4_v2_paper_like_thumb_calibrated` | 0.580 | 0.528 | 1.078 | False | 1.370 | thumb |
| `can_cylinder` | `bare_msk` | 1.900 | 0.528 | 2.398 | False | 1.346 | thumb |
| `can_cylinder` | `D4_v2_hybrid_per_finger_calibrated` | 1.314 | 0.556 | 1.812 | False | 1.358 | index,middle,ring |
| `can_cylinder` | `D4_v2_paper_like_thumb_calibrated` | 0.846 | 0.540 | 1.344 | False | 1.346 | thumb |
| `apple_sphere` | `bare_msk` | 3.242 | 0.526 | 3.740 | False | 1.327 | thumb |
| `apple_sphere` | `D4_v2_hybrid_per_finger_calibrated` | 1.348 | 0.544 | 1.846 | False | 1.326 | ring |
| `apple_sphere` | `D4_v2_paper_like_thumb_calibrated` | 1.280 | 0.546 | 1.778 | False | 1.339 | thumb |

## Lift Baseline

| Condition | Contact (s) | Peak touch (s) | Lift (s) | Held? | dZ after lift (m) | Touch digits |
|---|---:|---:|---:|---|---:|---|
| `bare_msk` | 1.086 | 0.520 | 1.584 | False | -0.199 | thumb |
| `D4_v2_hybrid_per_finger_calibrated` | 1.228 | 0.598 | 1.726 | False | -0.207 | index,little,middle,ring,thumb |
| `D4_v2_paper_like_thumb_calibrated` | 0.580 | 0.528 | 1.078 | False | -0.199 | thumb |

## Readout

- The restored-best working `D4_v2` is the best current branch for multi-digit engagement.
- The paper-like branch still behaves as a mostly thumb-led grasp.
- These tables now reflect the restored-best `D4_v2` branch, without the later
  opposition-capture experiments on the main device path.
- None of the tested conditions yet converts contact into a true carried grasp
  after support removal.

## Next Step

The next controller pass should target **post-contact load sharing / hold
stability** on the working `D4_v2` branch, not more object geometry changes.
