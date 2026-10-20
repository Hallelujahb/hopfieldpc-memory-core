# Mini PC experiment protocol

`benchmarks/mini_pc_mnist.py` asks one question: does adding Hopfield memory to a predictive-coding
classifier improve accuracy, robustness to noise or training, once obvious alternative explanations
are controlled? This file fixes the protocol and the success criteria before any result is looked at.

## Model

One hidden layer, 784 -> 256 (tanh) -> 10, trained with PC inference learning: input and one-hot label
clamped, hidden state settled for 15 steps, then a weight update at the settled state. This is a small
standalone network that lets the coupling be tested before FabricPC integration. It is not a claim
about FabricPC.

## Data and selection

* MNIST train is split once (fixed seed) into 55,000 training and 5,000 validation images. The full
  10,000-image test set is used for every reported number.
* Training stops early on validation accuracy (patience 3, at most 30 epochs) and restores the best
  weights.
* Pull strength, beta and the readout beta are chosen on the validation images, corrupted with the same
  noise level as the test images. Test data is never used to choose anything.
* 5 seeds. Seeds change the initialization, the batch order and the noise draws. Tables report the mean
  and sample standard deviation over seeds, plus the paired per-seed difference from the baseline.

## Compared rows

| Row | Purpose |
|---|---|
| `pc_feedforward[train_lam=0]` | Baseline |
| `pc_feedforward[train_lam=0.1]` | Hopfield pull used during training inference, evaluated feed-forward |
| `pull[prototypes,dot]`, `pull[prototypes,l2]` | Test-time pull toward class-mean hidden prototypes |
| `pull[random,dot]`, `pull[random,l2]` | Control: random memory with the same norms. Does content matter? |
| `hidden_nearest_mean_l2`, `hidden_knn5` | Controls on the same hidden features |
| `hopfield_readout` | Soft-kNN Hopfield readout over stored hidden features |

No row adds trainable parameters at test time, so no parameter-matched control is needed.

## Success criteria (fixed before running)

1. A method counts as improving on the baseline only if, at some noise level, its mean paired difference
   from `pc_feedforward[train_lam=0]` is at least +0.01 accuracy and larger than twice the standard
   deviation of the paired differences over the 5 seeds.
2. A pull must also beat its `pull[random,*]` control under the same rule. Otherwise the gain does not
   come from the memory content.
3. The readout must beat `hidden_knn5` under the same rule to be called better than kNN. If it ties or
   loses, it is reported as equivalent to kNN.
4. The training-time pull is judged by comparing `pc_feedforward[train_lam=0.1]` with the baseline under
   rule 1, on clean and noisy inputs.
5. Null and negative results are reported with the same prominence as positive ones.

## Limitations

One hidden layer, one dataset, frozen prototypes built from training labels, a clean-trained network
(no noise augmentation), and a standalone implementation. The l2 score variant is new in this repo and
has no prior evidence here.
