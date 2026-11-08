import copy,json,math,subprocess,sys,tempfile,unittest
from pathlib import Path
from fractions import Fraction as F
from unittest.mock import patch
from boostlab.core import predict,logistic_loss
from boostlab.path import inspect,route
from boostlab.path_export import saved,build
from boostlab.demo import fixture,run_demo

class PathTests(unittest.TestCase):
    def test_complete_archive_reproduction(self):
        r=saved();self.assertEqual(build(),r);self.assertEqual(r['original_demo'],run_demo());self.assertEqual(len(r['arms']),3)
        self.assertEqual([a['report']['selection']['best_round'] for a in r['arms'].values()],[0,80,35])
    def test_all4860states_and_routes_against_independent_fraction(self):
        r=saved();count=0
        def leaf(tree,row):
            if 'value' in tree:return tree['value']
            return leaf(tree['left' if row[tree['feature']]<=tree['threshold'] else 'right'],row)
        for a in r['arms'].values():
            m=a['report']['last_fitted_model']
            for row,reference in zip(r['samples']['rows'],a['references']):
                actual=inspect(m,row);self.assertEqual(actual,reference);total=F();self.assertEqual(len(actual['trees']),80)
                for k,s in enumerate(actual['states']):
                    if k:total+=F.from_float(leaf(m['trees'][k-1],row))
                    expected=float(F.from_float(m['initial_logit'])+F.from_float(m['learning_rate'])*total)
                    self.assertAlmostEqual(s['logit'],expected,places=13);self.assertEqual(s['label'],int(s['logit']>=0));count+=1
                    p=predict(dict(m,trees=m['trees'][:k]),[row]);self.assertEqual(p['logits'][0],s['logit']);self.assertEqual(p['probabilities'][0],s['probability'])
        self.assertEqual(count,4860)
    def test_1458_edited_states_independent_fraction(self):
        r=saved();count=0
        for name,refs in r['edited_references'].items():
            m=r['arms'][name]['report']['last_fitted_model']
            for ref in refs:
                row=ref['row'];self.assertEqual(inspect(m,row),ref);total=F()
                for k,state in enumerate(ref['states']):
                    if k:
                        tree=m['trees'][k-1]
                        while 'value' not in tree:tree=tree['left' if row[tree['feature']]<=tree['threshold'] else 'right']
                        total+=F.from_float(tree['value'])
                    exact=float(F.from_float(m['initial_logit'])+F.from_float(m['learning_rate'])*total)
                    self.assertAlmostEqual(state['logit'],exact,places=13);count+=1
        self.assertEqual(count,1458)

    def test_all243metrics_independent_inference(self):
        r=saved();splits=dict(zip(['train','validation','test'],fixture()))
        def leaf(t,row):
            while 'value' not in t:t=t['left'] if row[t['feature']]<=t['threshold'] else t['right']
            return t['value']
        for a in r['arms'].values():
            m=a['report']['last_fitted_model'];self.assertEqual(len(a['history']),81)
            for h in a['history']:
                k=h['round']
                for name,s in splits.items():
                    z=[m['initial_logit']+m['learning_rate']*math.fsum(leaf(t,row) for t in m['trees'][:k]) for row in s['rows']]
                    # independent softplus form, finite fixture scores
                    loss=math.fsum(math.log1p(math.exp((1-2*y)*v))/len(z) for y,v in zip(s['labels'],z));acc=sum(int(v>=0)==y for y,v in zip(s['labels'],z))/len(z)
                    self.assertAlmostEqual(loss,h[name]['logloss'],places=13);self.assertEqual(acc,h[name]['accuracy'])
                training=a['report']['history'][k];self.assertAlmostEqual(training['train_logloss'],h['train']['logloss'],places=12);self.assertAlmostEqual(training['validation_logloss'],h['validation']['logloss'],places=12)
    def test_exact_threshold_tie_and_zero_prefix(self):
        m=dict(version='boostlab_v1',feature_names=['x'],initial_logit=0.,learning_rate=.5,trees=[dict(feature=0,threshold=1.,left=dict(value=-1.),right=dict(value=1.))])
        for x,left in [(1.,True),(math.nextafter(1.,math.inf),False),(math.nextafter(1.,-math.inf),True)]:
            p=inspect(m,[x],0);self.assertEqual(p['chosen']['label'],1);self.assertEqual(p['trees'][0]['steps'][0]['branch'],'left' if left else 'right');self.assertEqual(p['states'][1]['logit'],-.5 if left else .5)
    def test_invalid_nonmutation(self):
        a=saved()['arms']['depth_2']['report']['last_fitted_model'];old=copy.deepcopy(a);row=[.1,.2];copyrow=row[:];inspect(a,row,0);self.assertEqual(a,old);self.assertEqual(row,copyrow)
        for prefix in [-1,81,True,.5]:
            with self.assertRaises(ValueError):inspect(a,row,prefix)
        for row in [[],[True,1],['1',0],[float('nan'),0]]:
            with self.assertRaises(ValueError):inspect(a,row)
        bad=copy.deepcopy(a);bad['trees'][0]['threshold']=float('inf')
        with self.assertRaises(ValueError):inspect(bad,[0,0])
    def test_cli_strict_and_installed_report(self):
        r=saved();m=r['arms']['depth_1']['report']['last_fitted_model']
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'request.json';p.write_text(json.dumps(dict(model=m,row=[0,0],prefix=0)))
            q=subprocess.run([sys.executable,'-m','boostlab','path','--input',str(p)],capture_output=True,text=True);self.assertEqual(q.returncode,0,q.stderr);self.assertEqual(json.loads(q.stdout)['prefix'],0)
            p.write_text('{"model":{},"row":[],"row":[],"prefix":0}');q=subprocess.run([sys.executable,'-m','boostlab','path','--input',str(p)],capture_output=True,text=True);self.assertEqual(q.returncode,2);self.assertIn('duplicate',q.stderr);self.assertEqual(q.stdout,'')
