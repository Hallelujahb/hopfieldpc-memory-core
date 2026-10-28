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
* Method A of the proposal (`h_new = PC_update + lam * Retrieval(h)`) adds the retrieved vector, which
  has no energy behind it. Rewrite the energy-consistent step as
  `z <- (1 - eta*lam) z - eta dE_PC/dz + eta*lam retrieved(z)`. That is Method A with a shrink factor
  `(1 - eta*lam)` on the old state and retrieval weight `eta*lam`. So A is B with the decay term
  dropped, and B is the version with a descent guarantee.

## Which energy

The Hopfield term is `hopfield_energy(..., include_constants=True)`. `hopfield_energy` carries the query
term `1/2 |q|^2` in every mode, so its gradient is `q - retrieved(q)` and one unnormalized retrieval step is
a unit gradient step. The two constants, `(1/beta) log N` and `1/2 max |m_i|^2`, shift the value and not the
gradient, so they never change the dynamics. They are included here so the logged values match the full
modern Hopfield energy. On the 2-D example with memories (1,0), (0,1), (-1,0), beta 5 and query (0.8, 0.5),
the default value falls from -0.3953 to -0.4749 after the first retrieval step and the value with constants
falls from 0.3244 to 0.2448.

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
with Hessian-vector products. It returns the largest signed eigenvalue (a shifted power iteration), so a
strong negative direction does not hide the top one. The estimate is local, so repeat it if `z` moves a lot.

Scope: the rule is exact when `L_PC` is a true global bound on the PC Hessian, as in the linear chain
used in the tests. For a nonlinear PC network `L_PC` is local and the rule is a heuristic. It is
sufficient for descent, not tight in general.

Assumptions: fixed memory, unnormalized retrieval, and full-batch gradient steps on `z`.

## Several memory banks

A `MemoryBank` is a memory acting on a slice of the state with its own `beta` and `lam_b`. This follows the
layer-local memory idea of the proposal: one bank per layer instead of one memory that mixes incompatible
patterns. The Hopfield part of the Hessian is `sum_b lam_b H_b`, and each `H_b` is bounded by the identity on
its slice. So the Hopfield part is bounded by a diagonal matrix whose entry `i` is the sum of `lam_b` over
the banks that cover coordinate `i`, and its largest eigenvalue is

```
max_i  sum over banks b that cover i of  lam_b
```

Banks on different layers (disjoint slices) give `max_b lam_b`. Banks on the same slice give the sum.
Using the sum over all banks is also safe, but wastes step size when banks sit on different layers.
The same scope caveat applies: exact only when `L_PC` is a true global bound.

## Energy and retrieval traces

`settle` and `settle_banks` take `log_diagnostics=True` (default off, so the default output is
unchanged). They then record, at every step including the start:

* `entropy`: entropy of the softmax weights over memories. `log N` means a global average, near 0
  means one memory is retrieved.
* `topk_mass`: softmax mass on the `top_k` memories.
* `retrieval_distance`: `|z - retrieved(z)|`, which equals the norm of `dE_H/dz`, so it is zero at a
  fixed point of retrieval.

With banks each trace has one column per bank. Together with the `pc`, `hopfield` and `total`
energy traces these cover the logging the proposal asks for. They reuse `attention_entropy`,
`topk_mass` and `retrieval_distance` from `diagnostics.py`. `hopfield_trace` computes the same
quantities at a single state.

## Toy study (mechanism check)

`python -m benchmarks.pc_hopfield_toy`, config in `configs/pc_hopfield_toy.yaml`. Linear chain with 6 hidden prototypes, hidden size 16, output size 4, beta 4. `L_PC = 2.433` is exact for this energy. The state starts at the forward-pass value (`E_PC = 0.953`, distance to the clean prototype 3.037).

| lam | safe eta | final E_PC | mean distance to clean prototype | nearest-memory acc |
|---|---|---|---|---|
| 0 | 0.411 | 0.463 | 2.812 | 1.000 |
| 0.05 | 0.403 | 0.472 | 2.682 | 1.000 |
| 0.1 | 0.395 | 0.497 | 2.564 | 1.000 |
| 0.3 | 0.366 | 0.684 | 2.181 | 1.000 |
| 1 | 0.291 | 1.515 | 1.437 | 1.000 |
| 3 | 0.184 | 2.883 | 0.733 | 1.000 |

Step-size rule, `eta = fraction * 2 / (L_PC + lam)`:

| lam | fraction | energy monotone |
|---|---|---|
| 0.3 | 0.5 | True |
| 0.3 | 0.95 | True |
| 0.3 | 1.05 | False |
| 0.3 | 1.5 | False |
| 1 | 0.5 | True |
| 1 | 0.95 | True |
| 1 | 1.05 | False |
| 1 | 1.5 | False |
| 3 | 0.5 | True |
| 3 | 0.95 | True |
| 3 | 1.05 | False |
| 3 | 1.5 | False |

The pull moves the state toward the prototype and costs PC energy, as the theory says. Nearest-memory accuracy is already 1.000 at lam = 0, so this toy shows no downstream benefit. It checks the mechanism and the step-size rule only.

## Limitations and open questions

* The tests use a quadratic `E_PC` (a linear chain). A nonlinear PC energy needs local curvature estimates.
* The memory is fixed. A learnable or refreshed memory changes the energy during training.
* Inference is full-batch. Banks act on slices of one joint state vector. A per-layer step size (preconditioning) could allow larger steps and is not done here.
* Related work (predictive-coding associative memories, Hopfield variants) still has to be compared against before claiming anything beyond the engineering of the coupling and the step-size rule.
