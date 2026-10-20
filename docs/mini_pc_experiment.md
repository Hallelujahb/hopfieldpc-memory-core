# Mini PC experiment: does a Hopfield term help a predictive-coding classifier?

`benchmarks/mini_pc_mnist.py` trains a standalone one-hidden-layer PC network
(`src/hopfieldpc/minipc.py`, 784 -> 256 tanh -> 10) with PC inference learning, then compares, with no
retraining and no added parameters:

| Row | What it tests |
|---|---|
| `pc_feedforward` | Baseline: the network's own prediction |
| `pull(lam, beta)` | Methods A/B at test time: hidden state pulled toward frozen class-mean prototypes |
| `hidden_nearest_mean_l2` | Control: nearest class mean on the same hidden features |
| `hidden_knn5` | Control: 5-NN on the same hidden features |
| `hopfield_readout(beta)` | Method C: soft-kNN Hopfield readout over stored hidden features |

## How to read the results

* A Hopfield row counts only if it beats `pc_feedforward` **and** the control rows computed on the
  same features, on mean over seeds with the spread reported.
* With the output free at test time the pull's fixed point is a convex blend
  `z1* = (mu1(x) + lam * retrieved(z1*)) / (1 + lam)`, i.e. shrinkage toward class prototypes. It can only
  help if the prototypes carry information the feed-forward state lacks (most plausible on noisy inputs).
  This is verified by `tests/test_minipc.py::test_pull_fixed_point_is_blend_of_forward_state_and_retrieved_memory`.
* No trainable parameters are added at test time (frozen prototypes and keys), so no parameter-matched
  control is needed for these rows. A learnable memory would need one.
* `--train-lam > 0` also applies the pull during training inference (memory refreshed each epoch).
  Treat it as a separate experiment: the "prior-injection" finding quoted in the project docs predicts
  that tethering PC latents can hurt.

## Limitations

One hidden layer, one dataset, small training budget, standalone (not FabricPC), frozen prototypes built
from training labels. A null or negative result is a valid outcome and should be reported as such.
