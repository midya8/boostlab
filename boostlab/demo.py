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
    for depth in (1,2):
        result=fit(payload,rounds=80,depth=depth,learning_rate=.5,min_leaf=5,patience=80)
        model=result['model']
        arms['depth_'+str(depth)]={'selected_rounds':result['selection']['best_round'],
                                 'fitted_rounds':result['selection']['fitted_rounds'],
                                 'retained_nodes':sum(nodes(t) for t in model['trees']),
                                 'last_checkpoint_nodes':sum(nodes(t) for t in result['last_fitted_model']['trees']),
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
                    'Both depths use the same 80-round cap/rate/min_leaf/patience=80, not equal node capacity. '
                    'Descriptive development comparison, not causal attribution, held-out external benchmark '
                    'or proof that depth 2 is generally better. No hyperparameter sweep was used for this fixture.'}
