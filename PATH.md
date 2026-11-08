# All fitted trees, without a new selection

BoostLab v0.2 adds an offline live inspector to the existing synthetic experiment. The original first-order/Newton training code, data generator, budgets and model-selection rule are unchanged. Three original arms still fit 80 rounds with rate 0.5, min_leaf 5 and patience 80. Selected prefixes remain 0, 80 and 35. The source code and original README's selected/last-checkpoint numbers remain the reference, not a new comparison.

## Frozen path

For each input x and fitted trees T1,...,T80, a prefix k has

```
z_k = initial_logit + learning_rate * sum_(i=1..k) T_i(x)
p_k = sigmoid(z_k)
label_k = int(z_k >= 0)
```

A zero prefix is the original smoothed training-prevalence logit, not a missing result. Each tree route records feature, input, stored threshold, branch and leaf. Exact equality goes left. The stored threshold is the lower observed value, not a rounding-sensitive midpoint. Contributions are per-tree algebraic terms, not feature importance, causal attribution, confidence or a claim that each addition improved every metric.

The path API validates the complete existing model with the inference surface before traversing it. It retains every fitted tree and every prefix state even when inspecting the selected zero-tree stump. `prefix` only chooses the displayed state. Nothing is fitted or selected on the page. Original labels are shown only for the first 20 unaltered test inputs in source order. Editing either coordinate drops that label; reset restores it. Invalid inputs clear the current witness and forbid export. The browser limits editable coordinates to [-1e6,1e6] for presentation; the native model schema keeps its existing finite binary64 bound.

## Metrics and honest axes

All 243arm/prefix states include train, validation and report-only test logloss/accuracy. The test curve was not supplied to fitting or selection. The fixture was inspected during original development and patience changed previously; none of this becomes untouched evaluation by displaying more checkpoints. Prefix selection can itself overfit a finite validation set. No confidence intervals, external evaluation, calibration claim or causal isolation is supplied.

The logloss axis starts at zero and uses the same maximum across all 81prefixes and all three arms. The per-input logit path includes zero, with its own sign-aware scale. Both charts mark inspected and original-selected prefixes separately. Last/selected/zero buttons do not change the archived decision. Independent input edits affect the prediction/path, not the fixed-fixture metric curves. All discarded prefixes stay available, including the last stump's better accuracy but worse logloss versus the constant.

Equal rounds are not equal stored nodes or effective capacity. Mean and capped regularized Newton leaves differ in more than curvature. Newton uses residual-SSE split selection, L2=1 and unshrunk leaf cap 3; it has no Hessian-gain split search or line search. No XGBoost/LightGBM/sklearn equivalence.

## Summation is a numerical boundary

Original fitting updates each training logit incrementally. Existing inference instead uses `initial + rate*math.fsum(leaves)`. The archive uses that existing inference convention for every prefix. Independently recomputed losses agree with original histories within 1e-12, not necessarily bit-for-bit. The browser uses Neumaier-style compensated scalar summation, no native prediction code. Tests compare logits/probabilities at 2e-13*(1+abs(reference)); fixture labels must match exactly. Tiny exact-rational leaf sums independently check the represented coefficients. These are numerical comparisons, not exact real arithmetic, a rigorous forward-error certificate or a promise for near-zero threshold labels on arbitrary platforms. No original tolerance is weakened.

## Reproduce

```
python -m boostlab.path_export
python -m boostlab path-report
python -m boostlab path --input examples/path.json
python -m unittest discover -s tests -v
node --test tests/*.cjs
python -m pip wheel . --no-deps -w /tmp/wheels
python -m pip install --target /tmp/installed /tmp/wheels/boostlab-0.2.0-py3-none-any.whl
bash examples/verify_wheel.sh /tmp/installed
```

The archive stores complete models, original fit histories and inference metrics in three small arm files. Twelve bounded chunks retain60full native paths; three edited-input chunks retain 18 additional complete paths; one meta file retains the original demo summary. All JSON chunks are below 200KB. The exporter reproduces all three original 80-round fits and all contents without changing algorithm or selecting a new configuration. The installed package includes every artifact and all 7 modules; all 5 main commands execute outside checkout and the exporter API rebuilds the full archive. Duplicate keys now refuse at every JSON CLI input, rather than silently overwrite fields. Existing strict model v1/v2 schemas are unchanged.

## Independent verification

- 4,860 sample/prefix states (3 arms x 20 first test inputs x 81 prefixes), every logit, probability and label,60 complete tree-route witnesses. The native reference includes independent exact Fraction sums and separate recursive tree traversal.
- All 243 arm/prefix metric states, all original 256/128/512 rows in each split. A separate scalar inference/softplus calculation checks loss/accuracy without the archived metric helper.
- 18 edited paths and 1,458 prefix states, zero/boundary/extreme finite queries, exact threshold ties and adjacent binary64 values, constant prefix, strict invalid schemas and nonmutation.
- Actual Chrome all 4,860 states and 60 routes; 180 arm/input/round control states, physical slider key, original label dropped after edit, invalid export refusal, blocked/successful clipboard, actual JSON Blob and expanded 320/390/1280 layouts.
- All 29 original tests remain, including training independence from validation labels, exact arithmetic split controls, no test field, v1/v2 limits, negative selection gate and original deterministic demo.

## Sources and scope

- [gbm guide](https://gbm-developers.github.io/gbm/articles/gbm.html): established negative-gradient tree fitting and Bernoulli ratio. Our shrinkage/regularization/cap implementation is intentionally narrower.
- [Natekin and Knoll, 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3885826/): additive fitting, shrinkage and validation regularization. No empirical result is transferred.
- [scikit-learn early-stopping example](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_early_stopping.html): validation monitoring and patience context, not library use or same stopping mechanics.
- [Hands-On Machine Learning with R, gradient boosting](https://bradleyboehmke.github.io/HOML/gbm.html): depth/shrinkage/round-count tradeoffs. Our fixed fixture does not isolate their effects.
- [Python math.fsum](https://docs.python.org/3/library/math.html): accurate floating sums, not an exact-real guarantee.
- [Python floating-point tutorial](https://docs.python.org/3/tutorial/floatingpoint.html): representation and operation-order limits.

Original Friedman 2001citation remains in README/DESIGN.md. Established gradient boosting, original educational code. No paid dependency, private data, telemetry, uploads or remote model. MIT; attribution is not endorsement.
