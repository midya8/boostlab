import copy
import io
import json
import math
import unittest
from contextlib import redirect_stdout,redirect_stderr
from unittest.mock import patch
from boostlab import fit,predict
from boostlab.core import _newton_refine,sigmoid
from boostlab.cli import main
from boostlab.demo import run_demo
from test_core import data


class TestNewton(unittest.TestCase):
    def test_logistic_hessian_finite_difference(self):
        for z in (-10,-2,0,2,10):
            eps=1e-5;actual=(sigmoid(z+eps)-sigmoid(z-eps))/(2*eps)
            self.assertAlmostEqual(actual,sigmoid(z)*(1-sigmoid(z)),places=10)

    def test_manual_leaf_refinement_without_mutation(self):
        t={'feature':0,'threshold':0,'left':{'value':-.5},'right':{'value':.5}};before=copy.deepcopy(t)
        r,counts=_newton_refine(t,[[0],[1]],[-.5,.5],[.25,.25],0,3)
        self.assertEqual(r['left']['value'],-2);self.assertEqual(r['right']['value'],2)
        self.assertEqual(counts,{'clipped':0,'zero_denominator':0});self.assertEqual(t,before)

    def test_regularization_clipping_and_zero_denominator(self):
        leaf={'value':.5};x=[[0],[1]];g=[.5,.5];h=[.25,.25]
        r,c=_newton_refine(leaf,x,g,h,.5,3);self.assertEqual(r['value'],1)
        r,c=_newton_refine(leaf,x,g,h,0,.75);self.assertEqual(r['value'],.75);self.assertEqual(c['clipped'],1)
        r,c=_newton_refine(leaf,x,g,[0,0],0,3);self.assertEqual(r['value'],0);self.assertEqual(c['zero_denominator'],1)

    def test_subnormal_hessian_clips_before_division_overflow(self):
        r,c=_newton_refine({'value':1},[[0]],[1],[5e-324],0,3)
        self.assertEqual(r['value'],3);self.assertEqual(c['clipped'],1)

    def test_v1_and_v2_json_prediction_roundtrip(self):
        for mode in ('mean','newton'):
            r=fit(data(),rounds=6,patience=6,min_leaf=2,leaf_mode=mode)
            expected='boostlab_v1' if mode=='mean' else 'boostlab_v2'
            self.assertEqual(r['model']['version'],expected)
            self.assertEqual(predict(r['model'],[[1],[19]]),predict(json.loads(json.dumps(r['model'])),[[1],[19]]))
        self.assertIn('newton_leaves',r['history'][1])

    def test_newton_parameter_validation(self):
        for kw in [{'leaf_mode':'other'},{'leaf_mode':'newton','l2':-1},{'leaf_mode':'newton','l2':True},
                   {'leaf_mode':'newton','max_leaf_step':0},{'leaf_mode':'newton','max_leaf_step':math.inf},
                   {'l2':1},{'max_leaf_step':2}]:
            with self.subTest(kw=kw),self.assertRaises(ValueError):fit(data(),rounds=2,**kw)

    def test_v2_artifact_bound_and_training_metadata(self):
        m=fit(data(),rounds=2,patience=2,leaf_mode='newton')['model']
        for key,value in [('leaf_mode','mean'),('l2',-1),('max_leaf_step',0),('trees',[{'value':4}])]:
            bad=copy.deepcopy(m);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):predict(bad,[[1]])
        bad=copy.deepcopy(m);bad.pop('l2')
        with self.assertRaises(ValueError):predict(bad,[[1]])
        m['trees']=[{'value':2}];self.assertEqual(len(predict(m,[[1]])['logits']),1)

    def test_newton_training_stays_independent_of_validation_labels(self):
        a=data();b=copy.deepcopy(a);b['validation']['labels']=[1-y for y in b['validation']['labels']]
        kwargs={'rounds':5,'patience':5,'min_leaf':2,'leaf_mode':'newton','l2':1,'max_leaf_step':3}
        x=fit(a,**kwargs);y=fit(b,**kwargs)
        self.assertEqual(x['last_fitted_model'],y['last_fitted_model'])
        self.assertNotEqual(x['selection']['best_round'],y['selection']['best_round'])

    def test_newton_cli_and_wrong_mode_error_no_partial_json(self):
        out=io.StringIO()
        with patch('sys.stdin',io.StringIO(json.dumps(data()))),redirect_stdout(out):
            self.assertEqual(main(['train','--input','-','--rounds','2','--leaf-mode','newton','--l2','1']),0)
        self.assertEqual(json.loads(out.getvalue())['model']['version'],'boostlab_v2')
        out,err=io.StringIO(),io.StringIO()
        with patch('sys.stdin',io.StringIO(json.dumps(data()))),redirect_stdout(out),redirect_stderr(err):
            self.assertEqual(main(['train','--input','-','--l2','1']),2)
        self.assertEqual(out.getvalue(),'');self.assertIn('only to newton',err.getvalue())

    def test_demo_newton_arm_and_mean_control_unchanged(self):
        r=run_demo();n=r['arms']['newton_depth_2'];m=r['arms']['depth_2']
        self.assertEqual(n['selected_rounds'],35);self.assertEqual(n['retained_nodes'],237)
        self.assertAlmostEqual(n['test']['accuracy'],.900390625)
        self.assertAlmostEqual(n['test']['logloss'],.33412424245601263)
        self.assertAlmostEqual(m['test']['logloss'],.340021294809901)
        self.assertGreater(n['last_checkpoint_test']['logloss'],n['test']['logloss'])
