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

## The energy has to be the full one

`hopfield_energy` with the default `include_constants=False` leaves out `1/2 |q|^2`. That term depends
on q, so it is not a constant, and the default value is not the energy that retrieval descends. On the
2-D example with memories (1,0), (0,1), (-1,0), beta 5 and query (0.8, 0.5), the default value rises
after the first retrieval step (-0.8403 to -0.8253), while the full energy falls (0.3244 to 0.2448).
This module always uses the full energy, `hopfield_energy(..., include_constants=True)`.

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

Banks on different layers (disjoint slices) give `max_b lam_b`. Banks on the same slice give the sum.
Using the sum over all banks is also safe, but wastes step size when banks sit on different layers.
The same scope caveat applies: exact only when `L_PC` is a true global bound.

## Limitations and open questions

* The tests use a quadratic `E_PC` (a linear chain). A nonlinear PC energy needs local curvature estimates.
* The memory is fixed. A learnable or refreshed memory changes the energy during training.
* Inference is full-batch. Banks act on slices of one joint state vector. A per-layer step size (preconditioning) could allow larger steps and is not done here.
* Related work (predictive-coding associative memories, Hopfield variants) still has to be compared against before claiming anything beyond the engineering of the coupling and the step-size rule.
