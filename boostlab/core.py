"""First-order Bernoulli gradient boosting with deterministic residual trees."""
import math


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def sigmoid(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def logistic_loss(y, z):
    return max(z, 0) - y*z + math.log1p(math.exp(-abs(z)))


def _names(names):
    if not isinstance(names, list) or not 1 <= len(names) <= 16 or any(
            not isinstance(v, str) or not v.strip() for v in names) or len(set(names)) != len(names):
        raise ValueError('feature_names must be 1-16 distinct nonempty strings')


def _rows(rows, width):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 2000 or any(
            not isinstance(r, list) or len(r) != width or any(not _finite(v) for v in r) for r in rows):
        raise ValueError('rows must be 1-2000 finite aligned numeric lists')
    return [[float(v) for v in r] for r in rows]


def _split(data, width):
    if not isinstance(data, dict) or set(data) != {'rows', 'labels'}:
        raise ValueError('split requires rows and labels exactly')
    x = _rows(data['rows'], width); y = data['labels']
    if not isinstance(y, list) or len(y) != len(x) or any(type(v) is not int or v not in (0, 1) for v in y):
        raise ValueError('labels must be aligned integer 0/1 values')
    return x, list(y)


def _fit_tree(x, targets, depth, min_leaf):
    """Greedy SSE splits; lower observed value is the threshold, not a midpoint."""
    total = math.fsum(targets); n = len(targets)
    leaf = {'value': total / n}
    if all(v == targets[0] for v in targets):
        return {'value': targets[0]}
    if depth == 0 or n < 2*min_leaf:
        return leaf
    best = None; best_gain = 1e-14
    for j in range(len(x[0])):
        order = sorted(range(n), key=lambda i: x[i][j])
        left_sum = 0.
        for k in range(1, n):
            left_sum += targets[order[k-1]]
            if k < min_leaf or n-k < min_leaf or x[order[k-1]][j] == x[order[k]][j]:
                continue
            right_sum = total - left_sum
            gain = left_sum*left_sum/k + right_sum*right_sum/(n-k) - total*total/n
            if gain > best_gain:
                best_gain = gain; best = (j, x[order[k-1]][j], order[:k], order[k:])
    if best is None:
        return leaf
    j, threshold, left, right = best
    return {'feature': j, 'threshold': threshold,
            'left': _fit_tree([x[i] for i in left], [targets[i] for i in left], depth-1, min_leaf),
            'right': _fit_tree([x[i] for i in right], [targets[i] for i in right], depth-1, min_leaf)}


def _tree_value(tree, row):
    while 'value' not in tree:
        tree = tree['left'] if row[tree['feature']] <= tree['threshold'] else tree['right']
    return tree['value']


def _validate_tree(tree, width, remaining=3):
    if not isinstance(tree, dict):
        raise ValueError('tree must be an object')
    if set(tree) == {'value'}:
        if not _finite(tree['value']) or not -1 <= tree['value'] <= 1:
            raise ValueError('residual-tree leaf must be finite in [-1,1]')
        return
    if remaining <= 0 or set(tree) != {'feature', 'threshold', 'left', 'right'}:
        raise ValueError('invalid tree shape or depth exceeds 3')
    if type(tree['feature']) is not int or not 0 <= tree['feature'] < width or not _finite(tree['threshold']):
        raise ValueError('invalid split index/threshold')
    _validate_tree(tree['left'], width, remaining-1)
    _validate_tree(tree['right'], width, remaining-1)


def predict(model, rows):
    if not isinstance(model, dict) or set(model) != {'version', 'feature_names', 'initial_logit', 'learning_rate', 'trees'}:
        raise ValueError('invalid model schema')
    if model['version'] != 'boostlab_v1':
        raise ValueError('unsupported model version')
    _names(model['feature_names']); width = len(model['feature_names'])
    rate = model['learning_rate']; initial = model['initial_logit']; trees = model['trees']
    if not _finite(rate) or not 0 < rate <= 1 or not _finite(initial):
        raise ValueError('model requires finite initial logit and learning_rate in (0,1]')
    if not isinstance(trees, list) or len(trees) > 200:
        raise ValueError('model requires at most 200 residual trees')
    for tree in trees:
        _validate_tree(tree, width)
    x = _rows(rows, width)
    logits = [initial + rate*math.fsum(_tree_value(t, row) for t in trees) for row in x]
    if any(not math.isfinite(v) for v in logits):
        raise ValueError('prediction exceeds finite float range')
    return {'logits': logits, 'probabilities': [sigmoid(v) for v in logits]}


def _losses(labels, logits):
    return math.fsum(logistic_loss(y, z)/len(labels) for y, z in zip(labels, logits))


def fit(payload, rounds=80, depth=2, learning_rate=.3, min_leaf=5, patience=10):
    """Fit trees on training residuals; select only the validation-best prefix.

    Fixed-rate first-order updates, not Newton leaves or per-round line search.
    No test data field, tuning sweep, sample weights or missing values.
    """
    if not isinstance(payload, dict) or set(payload) != {'feature_names', 'train', 'validation'}:
        raise ValueError('training input requires feature_names/train/validation only')
    _names(payload['feature_names']); width = len(payload['feature_names'])
    x, y = _split(payload['train'], width); vx, vy = _split(payload['validation'], width)
    for name, value, lo, hi in [('rounds', rounds, 1, 200), ('depth', depth, 1, 3),
                                ('min_leaf', min_leaf, 1, 1000), ('patience', patience, 1, 200)]:
        if type(value) is not int or not lo <= value <= hi:
            raise ValueError(name + ' must be an integer in [' + str(lo) + ',' + str(hi) + ']')
    if not _finite(learning_rate) or not 0 < learning_rate <= 1:
        raise ValueError('learning_rate must be finite in (0,1]')
    pos = sum(y); initial = math.log((pos+.5)/(len(y)-pos+.5))
    z = [initial]*len(y); vz = [initial]*len(vy); trees = []
    history = [{'round': 0, 'train_logloss': _losses(y, z), 'validation_logloss': _losses(vy, vz)}]
    best_loss = history[0]['validation_logloss']; best_round = 0
    for stage in range(1, rounds+1):
        residuals = [label-sigmoid(logit) for label, logit in zip(y, z)]
        tree = _fit_tree(x, residuals, depth, min_leaf); trees.append(tree)
        z = [v+learning_rate*_tree_value(tree, row) for v, row in zip(z, x)]
        vz = [v+learning_rate*_tree_value(tree, row) for v, row in zip(vz, vx)]
        loss = _losses(vy, vz)
        history.append({'round': stage, 'train_logloss': _losses(y, z), 'validation_logloss': loss})
        if loss < best_loss-1e-12:
            best_loss = loss; best_round = stage
        if stage-best_round >= patience:
            break
    model = {'version': 'boostlab_v1', 'feature_names': list(payload['feature_names']),
             'initial_logit': initial, 'learning_rate': learning_rate, 'trees': trees[:best_round]}
    last_model = dict(model, feature_names=list(model['feature_names']), trees=list(trees))
    return {'model': model, 'last_fitted_model': last_model, 'history': history,
            'selection': {'best_round': best_round, 'fitted_rounds': len(trees),
                          'validation_logloss': best_loss, 'patience': patience,
                          'rule': 'earliest prefix improving validation logloss by more than 1e-12'},
            'scope': 'Deterministic first-order Bernoulli boosting with greedy SSE residual trees and fixed shrinkage. '
                     'Half-count smoothing for training-only initial prevalence. Validation selects the prefix, '
                     'including zero trees. No Newton leaf updates, line search, external data, test-based selection, '
                     'missing-value handling, causal claims or production-performance guarantee.'}
