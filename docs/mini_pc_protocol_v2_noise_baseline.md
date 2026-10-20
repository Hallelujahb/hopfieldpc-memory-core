# Mini PC experiment v2: noise-trained baseline

Written and committed before the v2 run. v1 (docs/mini_pc_protocol.md) is unchanged.

## Question

In v1 the baseline was trained on clean images and collapsed under noise (0.276 at noise 0.3).
Does the Hopfield pull (l2 score, class-prototype memory) still help a baseline that was trained
on noisy inputs?

## What changes, and only this

* A second PC network (same architecture, same seed and initialization) is trained with noise
  augmentation: each training batch is corrupted with sigma drawn uniformly from {0, 0.3, 0.5}.
  Noise 1.0 is never seen in training.
* Early stopping for that network uses a fixed noisy validation set (one third each at 0, 0.3, 0.5).
* Everything else is as in v1: split, seeds, architecture, optimizer, pull grid, selection of lam and
  beta on validation data corrupted at the test noise level, 5 seeds, full test set.
* The clean-trained network is retrained with the v1 code and evaluated on the same noisy test
  inputs, as a consistency check against v1.

## Rows

| Row | Purpose |
|---|---|
| `clean_feedforward` | v1 baseline, retrained. Must match v1 numbers. |
| `noisetrained_feedforward` | New baseline |
| `noisetrained_pull[prototypes,l2]` | Pull toward class-mean hidden prototypes |
| `noisetrained_pull[random,l2]` | Control: random memory with the same norms |
| `noisetrained_hidden_nearest_mean_l2`, `noisetrained_hidden_knn5` | Context controls on the same features |

Dot scoring is not run again, since it showed no effect in v1.

## Success criteria (fixed before running)

Same rule as v1: a paired difference counts only if its mean is at least +0.01 and larger than twice
the standard deviation of the paired differences over 5 seeds.

1. The pull is useful for a noise-trained baseline only if `noisetrained_pull[prototypes,l2]` beats
   `noisetrained_feedforward` at one or more of noise 0.3, 0.5, 1.0, AND beats
   `noisetrained_pull[random,l2]` at the same level.
2. If it does not, the v1 gain is reported as largely due to a fragile baseline.
3. The gain of `noisetrained_feedforward` over `clean_feedforward` is reported either way.
4. Null and negative results get the same prominence as positive ones.

## Limitations

Same grid as v1, whose upper edge (lam = 1) was often chosen. Extending the grid is a separate,
labelled exploratory run. Training noise matches the test noise type, which favours the new
baseline at 0.3 and 0.5 by design. Noise 1.0 tests extrapolation.
