import json
import math
import random
from .core import fit, predict, logistic_loss


def fixture():
    rng=random.Random(7)
    def make(n):
        x=[];y=[]
        for _ in range(n):
            a,b=rng.uniform(-2,2),rng.uniform(-2,2)
            label=int(a*b>0)
            if rng.random()<.05:label=1-label
            x.append([a,b]);y.append(label)
        return {'rows':x,'labels':y}
    return make(256),make(128),make(512)


def metrics(model, split):
    result=predict(model,split['rows']);n=len(split['labels'])
    return {'accuracy':sum(int(z>=0)==y for z,y in zip(result['logits'],split['labels']))/n,
            'logloss':math.fsum(logistic_loss(y,z)/n for y,z in zip(split['labels'],result['logits']))}


def nodes(tree):
    return 1 if 'value' in tree else 1+nodes(tree['left'])+nodes(tree['right'])


def run_demo():
    train,val,test=fixture();payload={'feature_names':['a','b'],'train':train,'validation':val}
    arms={}
    for name,depth,mode,l2 in [('depth_1',1,'mean',0),('depth_2',2,'mean',0),
                                ('newton_depth_2',2,'newton',1)]:
        result=fit(payload,rounds=80,depth=depth,learning_rate=.5,min_leaf=5,patience=80,
                   leaf_mode=mode,l2=l2,max_leaf_step=3)
        model=result['model']
        arms[name]={'selected_rounds':result['selection']['best_round'],
                                 'fitted_rounds':result['selection']['fitted_rounds'],
                                 'retained_nodes':sum(nodes(t) for t in model['trees']),
                                 'last_checkpoint_nodes':sum(nodes(t) for t in result['last_fitted_model']['trees']),
                                 'leaf_mode':mode,'l2':l2,
                                 'newton_leaves_clipped':sum(h.get('newton_leaves',{}).get('clipped',0) for h in result['history']),
                                 'newton_zero_denominators':sum(h.get('newton_leaves',{}).get('zero_denominator',0) for h in result['history']),
                                 'last_checkpoint_test':metrics(result['last_fitted_model'],test),
                                 'train':metrics(model,train),'validation':metrics(model,val),
                                 'test':metrics(model,test)}
        assert predict(model,test['rows'])==predict(json.loads(json.dumps(model)),test['rows'])
        assert result['selection']['validation_logloss']<=result['history'][0]['validation_logloss']+1e-12
    initial=dict(model,trees=[])
    return {'seed':7,'sizes':{'train':256,'validation':128,'test':512},
            'initial_constant':{'test':metrics(initial,test)},'arms':arms,
            'scope':'Selected synthetic XOR-sign fixture with 5% independently flipped labels. '
                    'Train builds trees, validation selects the best prefix, test is report-only. '
                    'All arms use the same 80-round cap/rate/min_leaf/patience=80, not equal node capacity. '
                    'Newton depth-2 additionally uses L2=1 and leaf step cap 3, on residual-SSE splits. '
                    'Descriptive development comparison, not causal attribution, held-out external benchmark '
                    'or proof that depth 2 or Newton leaves are generally better. No hyperparameter sweep was used for this fixture.'}
