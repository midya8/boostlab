import copy
import json
import math
import unittest
from boostlab import fit,predict
from boostlab.core import _fit_tree,_tree_value,sigmoid,logistic_loss


def data():
    rows=[[i] for i in range(20)];labels=[int(i>=10) for i in range(20)]
    return {'feature_names':['x'],'train':{'rows':rows,'labels':labels},
            'validation':{'rows':[[i+.1] for i in range(20)],'labels':list(labels)}}


class TestCore(unittest.TestCase):
    def test_bernoulli_negative_gradient_finite_difference(self):
        for y in (0,1):
            for z in (-20,-2,0,2,20):
                eps=1e-5;derivative=(logistic_loss(y,z+eps)-logistic_loss(y,z-eps))/(2*eps)
                self.assertAlmostEqual(derivative,sigmoid(z)-y,places=8)

    def test_stable_extreme_logloss_not_probability_clipping(self):
        self.assertEqual(logistic_loss(0,1000),1000);self.assertEqual(logistic_loss(1,-1000),1000)
        self.assertEqual(logistic_loss(1,1000),0);self.assertEqual(logistic_loss(0,-1000),0)
        self.assertEqual(sigmoid(-1000),0);self.assertEqual(sigmoid(1000),1)

    def test_hand_split_and_leaf_means(self):
        x=[[0],[1],[2],[3]];tree=_fit_tree(x,[-.5,-.5,.5,.5],1,1)
        self.assertEqual(tree,{'feature':0,'threshold':1,'left':{'value':-.5},'right':{'value':.5}})
        self.assertEqual([_tree_value(tree,r) for r in x],[-.5,-.5,.5,.5])

    def test_greedy_root_matches_bruteforce_sse(self):
        x=[[0,2],[1,0],[2,3],[3,1],[4,4],[5,2]];y=[-.3,.7,-.1,.5,-.8,.2]
        candidates=[]
        for j in range(2):
            for threshold in sorted(set(r[j] for r in x))[:-1]:
                a=[y[i] for i,r in enumerate(x) if r[j]<=threshold];b=[y[i] for i,r in enumerate(x) if r[j]>threshold]
                if min(len(a),len(b))<2:continue
                sse=sum((v-sum(a)/len(a))**2 for v in a)+sum((v-sum(b)/len(b))**2 for v in b)
                candidates.append((sse,j,threshold))
        expected=min(candidates);tree=_fit_tree(x,y,1,2)
        self.assertEqual((tree['feature'],tree['threshold']),expected[1:])

    def test_constant_features_targets_and_minleaf(self):
        self.assertEqual(_fit_tree([[1]]*6,[.2]*6,3,1),{'value':.2})
        self.assertEqual(_fit_tree([[1]]*4,[-.5,.5,-.5,.5],3,1),{'value':0})
        self.assertEqual(_fit_tree([[0],[1],[2]],[-.5,0,.5],3,2),{'value':0})

    def test_duplicate_values_keep_identical_rows_together(self):
        tree=_fit_tree([[0],[0],[1],[1]],[-.5,-.5,.5,.5],1,1)
        self.assertEqual(tree['threshold'],0)

    def test_feature_tie_is_deterministic(self):
        tree=_fit_tree([[0,0],[1,1],[2,2],[3,3]],[-.5,-.5,.5,.5],1,1)
        self.assertEqual(tree['feature'],0)

    def test_training_loss_decreases_and_roundtrip_inference(self):
        r=fit(data(),rounds=15,min_leaf=2,patience=15)
        losses=[v['train_logloss'] for v in r['history']]
        self.assertTrue(all(b<=a+1e-12 for a,b in zip(losses,losses[1:])))
        self.assertEqual(len(r['model']['trees']),r['selection']['best_round'])
        self.assertEqual(predict(r['model'],[[1],[15]]),predict(json.loads(json.dumps(r['model'])),[[1],[15]]))
        self.assertLess(predict(r['model'],[[1]])['probabilities'][0],.5)
        self.assertGreater(predict(r['model'],[[15]])['probabilities'][0],.5)

    def test_opposite_validation_retains_zero_tree_prefix_and_stops(self):
        p=data();p['validation']['labels']=[1-y for y in p['validation']['labels']]
        r=fit(p,rounds=20,min_leaf=2,patience=3)
        self.assertEqual(r['selection']['best_round'],0);self.assertEqual(r['selection']['fitted_rounds'],3)
        self.assertEqual(r['model']['trees'],[]);self.assertEqual(len(r['last_fitted_model']['trees']),3)

    def test_validation_changes_selection_not_tree_fitting(self):
        a=data();b=copy.deepcopy(a);b['validation']['labels']=[1-y for y in b['validation']['labels']]
        x=fit(a,rounds=5,min_leaf=2,patience=5);y=fit(b,rounds=5,min_leaf=2,patience=5)
        self.assertEqual(x['last_fitted_model'],y['last_fitted_model'])
        self.assertNotEqual(x['selection']['best_round'],y['selection']['best_round'])

    def test_initial_logit_training_only_half_count_smoothed(self):
        p=data();p['train']['labels']=[1]*20;r=fit(p,rounds=2,patience=2)
        self.assertEqual(r['model']['initial_logit'],math.log(20.5/.5))
        p['validation']['labels']=[1]*20;s=fit(p,rounds=2,patience=2)
        self.assertEqual(r['model']['initial_logit'],s['model']['initial_logit'])

    def test_no_input_mutation(self):
        p=data();before=copy.deepcopy(p);r=fit(p,rounds=3,patience=3)
        self.assertEqual(p,before);r['model']['feature_names'][0]='changed';self.assertEqual(p,before)

    def test_validation_schema_and_parameters(self):
        for kw in [{'rounds':True},{'depth':4},{'min_leaf':0},{'patience':0},{'learning_rate':math.nan},{'learning_rate':1.1}]:
            with self.subTest(kw=kw),self.assertRaises(ValueError):fit(data(),**kw)
        for which,key,value in [('train','labels',[True]*20),('validation','labels',[0]),('train','rows',[[math.inf]]*20)]:
            p=data();p[which][key]=value
            with self.assertRaises(ValueError):fit(p)
        p=data();p['test']={}
        with self.assertRaises(ValueError):fit(p)
        p=data();p['train']['rows'][0]=[10**1000]
        with self.assertRaises(ValueError):fit(p)

    def test_model_schema_validation(self):
        model=fit(data(),rounds=2,patience=2)['model']
        for key,value in [('version','v0'),('initial_logit',math.inf),('learning_rate',0),('trees',[{'value':2}]),
                          ('trees',[{'feature':True,'threshold':1,'left':{'value':0},'right':{'value':0}}])]:
            m=copy.deepcopy(model);m[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):predict(m,[[1]])
        with self.assertRaises(ValueError):predict(model,[[True]])
        with self.assertRaises(ValueError):predict(model,[])

    def test_excessive_tree_depth_rejected(self):
        model=fit(data(),rounds=1,patience=1)['model'];tree={'value':0}
        for _ in range(4):tree={'feature':0,'threshold':1,'left':tree,'right':{'value':0}}
        model['trees']=[tree]
        with self.assertRaises(ValueError):predict(model,[[1]])
