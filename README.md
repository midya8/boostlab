# BoostLab

Train a binary classifier with Bernoulli gradient boosting and
shallow residual trees, from scratch. Dependency-free Python 3.10+, stable
logit-space loss, validation-only prefix selection, versioned JSON inference,
29 tests and an offline demo. Not XGBoost, LightGBM or sklearn compatibility.

## A trained interaction model, with the unselected control visible

```sh
python -m boostlab demo
bash examples/demo.sh
```

Synthetic sign-of-product classification: independent uniform coordinates in
[-2,2], with independently flipped labels at probability 0.05. Seed 7 generates
256 training, 128 validation and 512 report-only test rows.

| Checkpoint | Rounds in checkpoint | Stored nodes | Test accuracy | Test logloss |
|---|---:|---:|---:|---:|
| Training-prevalence constant | 0 | 0 | 0.4980 | 0.7060 |
| Depth-1, validation-selected | 0 | 0 | 0.4980 | 0.7060 |
| Depth-1, last fitted 80-round checkpoint | 80 (not selected) | 240 | 0.6367 | 0.7366 |
| Depth-2, mean leaves, selected and last fitted | 80 | 556 | 0.8906 | 0.3400 |
| Depth-2, Newton leaves, validation-selected | 35 | 237 | 0.9004 | 0.3341 |
| Depth-2, Newton leaves, last fitted | 80 (not selected) | 538 | 0.8984 | 0.3469 |

All three arms actually fit 80 rounds with rate 0.5, min_leaf 5, patience 80.
The unselected full-budget stump checkpoint is shown, not hidden behind prefix
selection: it improves threshold accuracy but worsens logloss versus the
constant on this test fixture. Selection optimizes **validation logloss**, not
accuracy or test results. Depth-2 selected training/validation logloss is
0.2487/0.3457. Newton depth-2 uses L2=1 and maximum unshrunk leaf step 3;
its selected training/validation logloss is 0.1777/0.3379, with zero clipped or
zero-denominator leaves over all 80 rounds. Same round budgets are not equal
node capacity. These settings were chosen for the demo without a Newton grid
search; this is not a controlled isolation of curvature or regularization.

These are development-fixture results, not an external benchmark, statistical
significance result, causal isolation of depth, or universal estimator ranking.
The fixture deliberately contains an interaction. No hyperparameter sweep was
used. During development patience changed from 12 to 80 so both unselected
full-budget checkpoints would be visible; test scores never enter fit or prefix
selection. Do not read the fixture as an untouched final evaluation.

## What is implemented

For logit z, the per-row Bernoulli loss is
`max(z,0) - y*z + log1p(exp(-abs(z)))`. Its negative gradient is `y - sigmoid(z)`.
Each round fits a deterministic greedy squared-error tree to those residuals,
then adds `learning_rate * tree(x)` to the logit. Leaves predict mean residuals.

The default is unchanged **fixed-rate first-order** mean-residual leaves.
Optional `leaf_mode="newton"` refines each leaf of the same residual-SSE tree:
`sum(y-p) / (sum(p*(1-p)) + l2)`, bounded by `max_leaf_step` before shrinkage.
It does not change split gain to a Hessian objective, solve each leaf's exact
likelihood optimum, or use a line search. This is not an XGBoost clone.
Training history records clipped and zero-denominator leaf counts per round.
With a zero denominator the leaf returns zero, explicitly counted. With a
nonzero denominator the cap is checked before division to avoid overflow.
The training
initial logit is `log((positives + 0.5)/(negatives + 0.5))`, with explicit
half-count smoothing so single-class training sets remain finite.

Tree candidates sort each feature and test splits between distinct observed
values, respecting min_leaf. The stored threshold is the lower observed value,
not a midpoint that can round onto the wrong side for adjacent floats. Greedy
SSE gain must exceed 1e-14. Exactly tied gains retain the first feature/threshold
encountered. Constant residuals/features and too-small nodes remain leaves.
This is greedy, not a globally optimal tree search.

## Fit, select, serialize, predict

```python
from boostlab import fit, predict

report = fit({
    "feature_names": ["x"],
    "train": {"rows": [[0], [1], [2], [3]], "labels": [0, 0, 1, 1]},
    "validation": {"rows": [[0.1], [1.1], [2.1], [3.1]], "labels": [0, 0, 1, 1]},
}, rounds=20, min_leaf=1)
probabilities = predict(report["model"], [[0.5], [2.5]])["probabilities"]
```

`fit` accepts exactly feature_names/train/validation. Splits contain rows and
integer 0/1 labels. There is no test-data field. Training builds every tree;
validation affects only prefix selection and the patience stop, not fitted tree
values for a common prefix. The earliest prefix improving validation logloss by
more than 1e-12 is retained, including the zero-tree initial model. The returned
history includes round zero and every fitted round.

`model` is the selected inference artifact. `last_fitted_model` preserves the
last checkpoint for audit, not automatic deployment. Artifacts contain version
`boostlab_v1`, ordered feature names, initial_logit, learning_rate and trees for
mean leaves. Newton models use `boostlab_v2`, also recording leaf_mode, l2 and
max_leaf_step. Inference still accepts strict v1 artifacts, including their
[-1,1] leaf bound; v2 leaves are checked against the recorded cap.
Inference returns logits and probabilities without applying an external
calibrator. JSON round trips preserve predictions in the tests.

```sh
python -m boostlab train --input examples/train.json --rounds 6 --min-leaf 2
# Output is a report containing model, last_fitted_model, history and selection.
python -m boostlab train --input examples/train.json --leaf-mode newton --l2 1 --max-leaf-step 3
python -m boostlab predict --input prediction-request.json
# prediction-request.json is {"model": <report.model>, "rows": [[1], [10]]}
```

Both commands support `--input -`. Invalid input exits 2 with stderr and no
partial JSON output. The demo exercises CLI artifact transfer and a negative
validation gate that discards fitted trees instead of forcing an improvement.

## Bounds and limits

1-16 finite numeric features, 1-2000 rows per split/inference request, 1-200
rounds, depth 1-3, min_leaf 1-1000, patience 1-200, learning_rate in (0,1]. Rows
are converted to binary64 floats, so large integer distinctions can be lost.
No missing/categorical features, weights, multiclass, stochastic subsampling,
histograms, Hessian-gain split construction, feature importance or distributed work.
Newton L2 is finite in [0,1000000], step cap in (0,10]; defaults 0 and 3.
Nondefault L2/cap options are rejected in mean mode, rather than silently ignored.
No model download, network call or paid dependency.

Validation selection can overfit a small validation set, and the caller owns
split quality, feature provenance and hyperparameter choice. Overlapping people,
time leakage or shifted deployment data are not detected by the schema. No
confidence bounds, probability-calibration guarantee, causal conclusion or
production-performance promise. Node counts describe stored model size, not
walltime or memory speedup.

## Verification

```sh
python -m unittest discover -s tests -v
```

29 tests cover finite-difference Bernoulli gradients, extreme stable loss, hand
leaf/split arithmetic, a brute-force SSE root reference, min_leaf, constant and
duplicate features, deterministic ties, adjacent floats, training descent on a
separable fixture, validation-only prefix selection, a zero-tree rejected-model
gate, unchanged fitted trees when validation labels change, training-only
initial prevalence, single-class labels, strict artifacts, no input mutation,
CLI JSON round trips and deterministic demo metrics. CI runs Python 3.10-3.13
plus the offline demo. Newton checks add Hessian finite differences, hand leaf
arithmetic, L2, caps, zero/subnormal denominators, v1/v2 round trips, strict
artifact bounds, training independence from validation labels and CLI errors.

## Sources and credit

- [Friedman (2001), Greedy Function Approximation: A Gradient Boosting Machine](https://doi.org/10.1214/aos/1013203451)
- [gbm guide: negative-gradient tree fitting and Bernoulli formulas](https://gbm-developers.github.io/gbm/articles/gbm.html)

The gradient-boosting idea is standard. BoostLab deliberately omits the source
algorithms' line search. Its optional Newton leaves use the guide's standard
Bernoulli ratio with explicit L2 and a step cap; it is an independent
educational fixed-step implementation, not a reproduction of their full
algorithms or a new method. MIT license.
