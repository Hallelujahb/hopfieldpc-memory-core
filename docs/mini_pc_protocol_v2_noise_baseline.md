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

## Results

5 seeds, full 10,000-image test set, mean test accuracy. Paired differences and per-seed values are
in results.csv.

| method | noise 0 | noise 0.3 | noise 0.5 | noise 1.0 |
|---|---|---|---|---|
| clean_feedforward | 0.966 | 0.276 | 0.140 | 0.104 |
| noisetrained_feedforward | 0.951 | 0.931 | 0.892 | 0.690 |
| noisetrained_pull[prototypes,l2] | 0.951 | 0.931 | 0.892 | 0.691 |
| noisetrained_pull[random,l2] | 0.950 | 0.931 | 0.889 | 0.685 |
| noisetrained_hidden_nearest_mean_l2 | 0.866 | 0.801 | 0.651 | 0.390 |
| noisetrained_hidden_knn5 | 0.950 | 0.908 | 0.765 | 0.397 |

Against the criteria written above:

1. The pull does not beat the noise-trained baseline at any noise level (paired differences 0.000,
   0.000, 0.000, +0.001; the bar was +0.01). Criterion 1 fails.
2. The v1 gain is therefore reported as largely due to a fragile baseline.
3. Noise training itself gains about +0.65, +0.75 and +0.59 over the clean baseline at noise 0.3, 0.5
   and 1.0 (noise 1.0 was never seen in training), at a cost of about 0.015 on clean data.

Validation selected the weakest pull in most cases (lam = 0.05, beta = 0.05, also the first grid
entry, which wins ties), so within this grid no pull setting beat doing almost nothing.

Not claimed: that Hopfield memory cannot help predictive coding. This covers one hidden layer, one
dataset, Gaussian noise, a frozen class-prototype memory and this grid.
