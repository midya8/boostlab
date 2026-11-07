import io
import json
import math
import unittest
from contextlib import redirect_stdout,redirect_stderr
from unittest.mock import patch
from boostlab.cli import main
from boostlab.demo import run_demo
from boostlab.core import _fit_tree
from test_core import data


class TestCLI(unittest.TestCase):
    def test_train_predict_cli_roundtrip(self):
        out=io.StringIO()
        with patch('sys.stdin',io.StringIO(json.dumps(data()))),redirect_stdout(out):
            self.assertEqual(main(['train','--input','-','--rounds','3','--patience','3']),0)
        report=json.loads(out.getvalue());p={'model':report['model'],'rows':[[1],[19]]};out=io.StringIO()
        with patch('sys.stdin',io.StringIO(json.dumps(p))),redirect_stdout(out):
            self.assertEqual(main(['predict','--input','-']),0)
        r=json.loads(out.getvalue());self.assertEqual(len(r['logits']),2)
        self.assertLess(r['probabilities'][0],r['probabilities'][1])

    def test_cli_bad_input_no_partial_report(self):
        for command in ('train','predict'):
            for body in ('{','{}'):
                out,err=io.StringIO(),io.StringIO()
                with patch('sys.stdin',io.StringIO(body)),redirect_stdout(out),redirect_stderr(err):
                    self.assertEqual(main([command,'--input','-']),2)
                self.assertEqual(out.getvalue(),'');self.assertIn('input error',err.getvalue())

    def test_demo_is_measured_and_repeats(self):
        a=run_demo();b=run_demo();self.assertEqual(a,b)
        self.assertEqual(a['arms']['depth_2']['selected_rounds'],80)
        self.assertEqual(a['arms']['depth_1']['fitted_rounds'],80)
        self.assertEqual(a['arms']['depth_1']['selected_rounds'],0)
        self.assertAlmostEqual(a['arms']['depth_2']['test']['accuracy'],.890625)
        self.assertGreater(a['arms']['depth_1']['last_checkpoint_nodes'],0)

    def test_adjacent_float_split_uses_lower_observed_value(self):
        a=1.;b=math.nextafter(a,math.inf)
        tree=_fit_tree([[a],[b]],[-.5,.5],1,1)
        self.assertEqual(tree['threshold'],a)
        self.assertEqual(tree['left']['value'],-.5);self.assertEqual(tree['right']['value'],.5)
