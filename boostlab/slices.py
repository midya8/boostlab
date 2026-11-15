"""Complete binary64 coordinate slices of a fixed fitted tree ensemble.

This partitions one bounded coordinate while holding every other input fixed.
It does not retrain, choose a prefix, search other coordinates or certify a
multidimensional robustness radius. Thresholds from unreachable branches may
make the partition finer than necessary. No interpolation or sampling grid.
"""
import math

from .core import _finite, predict
from .path import inspect


def coordinate_slice(model, row, feature, lower, upper):
    """Return every nonempty threshold cell on [lower, upper], including ties.

    Input coordinates are binary64. Integer thresholds keep Python comparison
    semantics: the cut is the greatest binary64 value not above the threshold.
    Each cell records its closed representable endpoints, full route witness,
    logit, probability and first-class-threshold label (logit >= 0).
    """
    predict(model, [row])  # Owning inference surface checks schema and bounds.
    width = len(model['feature_names'])
    if type(feature) is not int or not 0 <= feature < width:
        raise ValueError('feature must be an in-range integer index')
    if not _finite(lower) or not _finite(upper):
        raise ValueError('slice endpoints must be finite numbers')
    lower, upper = float(lower), float(upper)
    if lower > upper:
        raise ValueError('lower must not exceed upper after binary64 conversion')
    cuts = set()

    def walk(tree):
        if 'value' in tree:
            return
        threshold = float(tree['threshold'])
        if threshold > tree['threshold']:
            threshold = math.nextafter(threshold, -math.inf)
        if tree['feature'] == feature and lower <= threshold < upper:
            cuts.add(threshold)
        walk(tree['left'])
        walk(tree['right'])

    for tree in model['trees']:
        walk(tree)
    cells = []
    start = lower
    for stop in sorted(cuts) + [upper]:
        if start > stop:
            continue
        edited = [float(x) for x in row]
        edited[feature] = start
        witness = inspect(model, edited)
        chosen = witness['chosen']
        cells.append({'lower': start, 'upper': stop,
                      'logit': chosen['logit'],
                      'probability': chosen['probability'],
                      'label': chosen['label'], 'witness': witness})
        start = math.nextafter(stop, math.inf)
    return {'feature': feature, 'feature_name': model['feature_names'][feature],
            'fixed_row': [float(x) for x in row], 'lower': lower, 'upper': upper,
            'thresholds': sorted(cuts), 'cells': cells,
            'scope': 'Complete bounded binary64 coordinate slice of this fixed '
                     'artifact. Other coordinates fixed. Not continuous-real '
                     'arithmetic, calibration, causal intervention, retraining '
                     'or multidimensional robustness. Partition need not be minimal.'}
