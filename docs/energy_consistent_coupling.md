# Energy-consistent coupling of Hopfield memory and predictive-coding inference

## Summary

Treat the Hopfield memory as one more term of the energy that PC inference already descends,
instead of a module that overwrites the hidden state:

```
E_total(z) = E_PC(z) + lam * E_H(z; M, beta)
z <- z - eta * ( dE_PC/dz + lam * (z - retrieved(z)) )
```

* `dE_H/dz = z - softmax(beta z M^T) M = z - retrieved(z)` (`hopfield_energy_grad`).
* `lam = 0` is plain PC, the baseline. With no PC term and `eta * lam = 1`, one step is exactly
  one-step Hopfield retrieval.
* Method A of the proposal (`h_new = PC_update + lam * Retrieval(h)`) adds the retrieved vector.
  The energy-consistent form pulls toward it, `lam * (retrieved - h)`. Written that way, Methods A
  and B are the same design: A is the gradient step of B.

## The energy has to be the full one

`hopfield_energy` with the default `include_constants=False` leaves out `1/2 |q|^2`. That term depends
on q, so it is not a constant, and the default value is not the energy that retrieval descends. On the
2-D example with memories (1,0), (0,1), (-1,0), beta 5 and query (0.8, 0.5), the default value rises
after the first retrieval step (-0.8403 to -0.8253), while the full energy falls (0.3244 to 0.2448).
Use `hopfield_full_energy` or `include_constants=True` for anything that relies on descent or
gradients. `tests/test_energy_descent.py` covers both.

## Stability rule

The Hessian of `E_H` is `I - beta * Cov_p(m)`, where `Cov_p` is the covariance of the memory vectors
under the softmax weights. Its eigenvalues lie in `[1 - beta * M_max^2, 1]`, so the upper curvature of
the Hopfield term is exactly 1 for every beta and every memory. Monotone descent needs only the upper
curvature, so for gradient-descent inference with a fixed memory:

```
eta < 2 / (L_PC + lam)          (eta <= 1 / (L_PC + lam) is the conservative choice)
```

`L_PC` is the largest Hessian eigenvalue of `E_PC`. `beta` and the memory norm do not enter the step-size
limit. They set the negative curvature, which is how deep and separated the valleys are (research
question 6 of the proposal). `estimate_top_curvature` estimates `L_PC` for any differentiable energy
with Hessian-vector products. The estimate is local, so repeat it if `z` moves a lot.

Assumptions: fixed memory, full-batch gradient steps on `z`, and the same scoring rule in retrieval and
in the energy.

## Score variants

`score="dot"` is standard modern Hopfield retrieval. `score="l2"` adds the bias `-1/2 |m_i|^2` to each
similarity, so memories are ranked by Euclidean distance. Dot-product scoring favours memories with a
large norm. The repo's own MNIST pixel benchmark shows the cost of that: argmax of the dot product with
class means reaches 0.624 clean accuracy, nearest class mean in L2 reaches 0.823. The l2 variant keeps
the gradient `z - retrieved(z)` and the same Hessian, so the stability rule above is unchanged
(`tests/test_coupling.py` checks both). It has not been evaluated on a task yet. Euclidean-similarity
Hopfield variants appear in Millidge et al., Universal Hopfield Networks (ICML 2022), so treat this as
engineering and a controlled comparison, not as a new memory model.

## Results of the toy study

`python -m benchmarks.pc_hopfield_toy`, config in `configs/pc_hopfield_toy.yaml`. Linear chain with
`d_in=32, d_h=16, d_o=4`, 6 prototypes, 200 noisy samples (`noise=0.5`), `beta=4`, `L_PC = 2.433`,
Hopfield curvature range `[-8, 1]`. The output is smaller than the hidden layer, so the clamped output
cannot pin down every hidden coordinate. This is a mechanism check, not evidence about MNIST or FabricPC.

| lam | final E_PC | final E_total | mean distance to clean prototype | nearest-memory acc |
|---|---|---|---|---|
| start (forward pass) | 0.9533 | - | 3.0373 | - |
| 0 | 0.4630 | 0.4630 | 2.8119 | 1.000 |
| 0.05 | 0.4724 | 0.6873 | 2.6820 | 1.000 |
| 0.1 | 0.4971 | 0.8938 | 2.5638 | 1.000 |
| 0.3 | 0.6839 | 1.5818 | 2.1808 | 1.000 |
| 1 | 1.5155 | 3.0678 | 1.4373 | 1.000 |
| 3 | 2.8833 | 5.0881 | 0.7331 | 1.000 |

The pull moves the hidden state toward the stored prototype and costs PC energy, so the network
explains its input less well. Nearest-memory accuracy is already 1.0 without the pull, so this toy shows
no downstream benefit.

Step-size rule, 300 inference steps: for `lam` in {0.3, 1, 3} the total energy decreases monotonically
at 0.5x and 0.95x of `2 / (L_PC + lam)` and diverges at 1.05x and 1.5x.

## Limitations and open questions

* The toy uses a quadratic `E_PC`. A nonlinear PC energy needs local curvature estimates.
* The memory is fixed. A learnable or refreshed memory changes the energy during training.
* Inference is full-batch. Per-layer memory banks and a joint energy over several layers are the next
  step: a sum of per-layer terms with one `lam_l` each.
* Related work to compare against before claiming anything beyond the engineering of the coupling and
  the step-size rule: Salvatori et al., Associative Memories via Predictive Coding (NeurIPS 2021), and
  Millidge et al., Universal Hopfield Networks (ICML 2022).
