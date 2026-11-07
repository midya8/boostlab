# BoostLab design

## Model and objective

Binary Bernoulli classification on finite numeric rows. The ensemble predicts a
logit, then applies a stable sigmoid. The loss is stable softplus minus y*z, not
clipped probability logloss. The per-row negative gradient is y-sigmoid(z),
checked by central finite differences at both labels and multiple logits.

The training-only intercept uses half-count smoothed prevalence. This is an
explicit numerical default, not an estimate from validation/test labels.
Each round fits a greedy squared-error regression tree to current residuals and
adds it with fixed shrinkage. Default leaves are mean residuals in [-1,1]. Optional Newton leaf
refinement uses sum(y-p)/(sum(p*(1-p))+L2), then caps the unshrunk step.
The residual-SSE split objective stays unchanged; no Hessian-gain tree
construction, exact likelihood optimization or line search. This is an educational variant,
not a compatible clone of gbm/sklearn/XGBoost or new boosting theory.

## Greedy tree mechanics

Sort candidate features stably and scan distinct-value boundaries. Prefix sums
compute SSE improvement from the two child means. Respect min_leaf on both
sides. Retain the first exactly tied gain; require gain above 1e-14. Thresholds
are lower observed values, preserving a valid boundary even for adjacent
binary64 floats. Constant targets return their identical float directly;
constant features or insufficient rows produce a mean leaf. Max depth 3 bounds
the tree shape, but greedy local choice does not optimize the global tree.

## Optional Newton leaf refinement

Use `leaf_mode="newton"`, finite L2 in [0,1000000] (default 0), and step cap
in (0,10] (default 3). Nondefault L2/cap are errors in default mean mode.
Training probabilities from the start of each round define both residuals and
Hessians. Route training rows through the fitted SSE tree, then replace each
leaf by its regularized ratio. No validation labels enter this calculation.
A zero denominator returns zero with a diagnostic count. Compare the numerator
against cap times denominator before dividing; this handles subnormal Hessians
without producing an infinite intermediate. Cap-hit counts are reported too.
A cap is an explicit numerical/training choice, not a convergence guarantee.
Hessians are checked by finite differences of sigmoid, and leaf arithmetic,
regularization, cap, zero and subnormal denominators have direct tests.

## Split ownership and prefix selection

Only training rows/labels determine the intercept, residuals, splits and leaf
values. Validation logloss chooses the earliest prefix improving by over 1e-12,
including round zero. Patience stops after that many rounds without a qualifying
improvement. Changing validation labels can change the chosen prefix or stop,
but not the shared fitted prefix itself, which is tested.

The API does not accept test data. A separate demo reports test metrics only
after fitting. This separation does not prove that a caller's feature pipeline,
split or hyperparameter choice is leakage-free. Both the selected model and
last fitted checkpoint are returned so validation rejection cannot hide the
trained baseline. History records all train/validation losses, including zero.

## Fixture and control transparency

Seed 7 generates 256/128/512 train/validation/test rows. Coordinates uniform in
[-2,2], class is sign of their product, independently flipped with probability
0.05. Depth 1 and 2 use 80 rounds, rate .5, min_leaf 5, patience 80. Same round
budget, not equal node capacity. Initial draft patience was 12; it became 80 to
show the full fitted checkpoint for both arms, not to select on test scores.
This is development data, not an untouched final benchmark.

Selected stumps retain round zero: test accuracy .4980/logloss .7060. Their
unselected 80-round checkpoint has 240 nodes, accuracy .6367/logloss .7366.
The depth-2 selected 80-round model has 556 nodes, accuracy .8906/logloss .3400;
training/validation losses .2487/.3457. Accuracy and logloss can disagree.
The additional Newton depth-2 arm uses the same round/rate/min_leaf/patience,
L2=1 and step cap 3. Validation selects round 35: 237 nodes, test accuracy
.900390625/logloss .3341242425, training/validation loss .1776606433/.3379099355.
Last round 80 has 538 nodes, test accuracy .8984375/logloss .3468631810.
No leaves clip or have zero denominators. This setting was not chosen from a
Newton grid search, but the fixture is development data and was already seen.
The comparison does not isolate a causal curvature or regularization benefit.

These are descriptive fixture results, not causal components, general model
rankings, calibration guarantees or a production-performance benchmark.

## Artifact and numerical boundary

Default version boostlab_v1, still accepted unchanged by prediction. Newton
version boostlab_v2 additionally records leaf_mode, l2 and max_leaf_step; strict
schema validation checks leaves against that cap instead of [-1,1]. Explicit ordered names, initial logit, rate and recursive
trees. Validate schemas, finite scalars, feature indices, residual leaf range,
depth and tree count before prediction. Inputs become binary64 floats, so large
integer distinctions can disappear. No NaN/Infinity or silent normalization.
Logit/probability predictions survive JSON round trips in tests. No dynamic
code, external model download, networking or automatic deployment.

Limits: 1-16 features, 1-2000 rows per request/split, 1-200 boosting rounds,
depth 1-3, min_leaf 1-1000, patience 1-200, rate in (0,1]. No missing values,
categorical encoding, weights, stochastic sampling, multiclass or histograms.

## Credit

- https://doi.org/10.1214/aos/1013203451
- https://gbm-developers.github.io/gbm/articles/gbm.html

Standard gradient boosting and Bernoulli residuals. Mean-residual leaves remain the default. Optional Bernoulli Newton ratios
follow the guide with explicit L2 and a cap. Residual-SSE splits and fixed rate
remain explicit simplifications, not the full source algorithms.
