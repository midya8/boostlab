import copy
import math
import random
import unittest

from boostlab import fit, predict
from boostlab.path import inspect
from boostlab.slices import coordinate_slice
from test_core import data


def model(trees):
    return {'version': 'boostlab_v1', 'feature_names': ['x', 'fixed'],
            'initial_logit': -.125, 'learning_rate': .5, 'trees': trees}


def split(feature, threshold, left, right):
    return {'feature': feature, 'threshold': threshold,
            'left': {'value': left}, 'right': {'value': right}}


class SliceTests(unittest.TestCase):
    def check(self, m, row, feature, lower, upper):
        before = copy.deepcopy((m, row))
        report = coordinate_slice(m, row, feature, lower, upper)
        self.assertEqual((m, row), before)
        cells = report['cells']
        self.assertEqual(cells[0]['lower'], float(lower))
        self.assertEqual(cells[-1]['upper'], float(upper))
        for index, cell in enumerate(cells):
            self.assertLessEqual(cell['lower'], cell['upper'])
            if index:
                self.assertEqual(cell['lower'], math.nextafter(cells[index-1]['upper'], math.inf))
            # Check endpoints, a bounded midpoint and representable neighbors.
            points = [cell['lower'], cell['upper'], cell['lower']/2 + cell['upper']/2,
                      math.nextafter(cell['lower'], math.inf),
                      math.nextafter(cell['upper'], -math.inf)]
            for point in points:
                if not cell['lower'] <= point <= cell['upper']:
                    continue
                edited = row[:]; edited[feature] = point
                expected = predict(m, [edited])
                self.assertEqual(cell['logit'], expected['logits'][0])
                self.assertEqual(cell['probability'], expected['probabilities'][0])
                self.assertEqual(cell['label'], int(expected['logits'][0] >= 0))
                # Branch choices and leaves must stay constant in each cell.
                actual = inspect(m, edited)
                for a, b in zip(actual['trees'], cell['witness']['trees']):
                    self.assertEqual(a['leaf'], b['leaf'])
                    self.assertEqual([x['branch'] for x in a['steps']],
                                     [x['branch'] for x in b['steps']])
        return report

    def test_threshold_ownership_adjacent_extreme_and_singleton(self):
        tiny = math.nextafter(0., math.inf)
        m = model([split(0, 0., -.5, .5), split(0, tiny, .25, -.25)])
        r = self.check(m, [0, 0], 0, -tiny, tiny*2)
        self.assertEqual([(c['lower'], c['upper']) for c in r['cells']],
                         [(-tiny, 0.), (tiny, tiny), (tiny*2, tiny*2)])
        self.check(m, [0, 0], 0, 0, 0)
        self.check(m, [0, 0], 0, -1.7976931348623157e308, 1.7976931348623157e308)
        for threshold in [2**53 + 1, 2**53 + 3, -(2**53 + 1), -(2**53 + 3)]:
            self.check(model([split(0, threshold, -.5, .5)]), [0, 0],
                       0, threshold - 8, threshold + 8)
        r = self.check(model([]), [0, 0], 0, -1, 1)
        self.assertEqual(len(r['cells']), 1)

    def test_other_coordinates_and_unreachable_cuts(self):
        tree = {'feature': 1, 'threshold': 0,
                'left': split(0, -.5, -.5, .5),
                'right': split(0, .5, .25, -.25)}
        m = model([tree, split(0, 0, .125, -.125)])
        for fixed in [-1, 0, 1]:
            r = self.check(m, [0, fixed], 0, -1, 1)
            self.assertEqual(r['thresholds'], [-.5, 0., .5])
        self.check(m, [0, 1], 1, -1, 1)

    def test_seeded_ensembles_and_both_actual_fit_modes(self):
        rng = random.Random(81)
        for _ in range(60):
            trees = [split(rng.randrange(2), rng.randrange(-8, 9)/8,
                           rng.randrange(-8, 9)/8, rng.randrange(-8, 9)/8)
                     for _ in range(6)]
            m = model(trees)
            for feature in [0, 1]:
                self.check(m, [.125, -.25], feature, -2, 2)
        for mode in ['mean', 'newton']:
            fitted = fit(data(), rounds=12, patience=12, min_leaf=2,
                         leaf_mode=mode, **({'l2': 1} if mode == 'newton' else {}))
            for key in ['model', 'last_fitted_model']:
                self.check(fitted[key], [0], 0, -10, 30)

    def test_every_archived_arm_selected_and_last_artifact(self):
        from boostlab.path_export import saved
        archive = saved()
        count = 0
        for arm in archive['arms'].values():
            for key in ['model', 'last_fitted_model']:
                for feature in [0, 1]:
                    self.check(arm['report'][key], [.125, -.25], feature, -2, 2)
                    count += 1
        self.assertEqual(count, 12)

    def test_validation_and_wrong_side_control(self):
        m = model([split(0, 0., -.5, .5)])
        left = predict(m, [[0., 0]])['logits'][0]
        right = predict(m, [[math.nextafter(0., math.inf), 0]])['logits'][0]
        self.assertNotEqual(left, right)
        for feature, lower, upper in [(True, 0, 1), (-1, 0, 1), (2, 0, 1),
                                     (0, True, 1), (0, 2, 1), (0, 0, math.inf),
                                     (0, math.nan, 1)]:
            with self.assertRaises(ValueError):
                coordinate_slice(m, [0, 0], feature, lower, upper)
        with self.assertRaises(ValueError):
            coordinate_slice(m, [0], 0, -1, 1)
