"""Frozen additive-prefix witnesses. No fitting, selection or calibration."""
import math
from .core import predict,sigmoid,_rows

def route(tree,row):
    steps=[];node=tree
    while 'value' not in node:
        j=node['feature'];t=node['threshold'];left=row[j]<=t
        steps.append(dict(feature=j,threshold=t,input=row[j],branch='left' if left else 'right',comparison='<= ' if left else '> '))
        node=node['left'] if left else node['right']
    return dict(steps=steps,leaf=node['value'])

def inspect(model,row,prefix=None):
    # Owning inference surface validates full schema, tree bounds and finite output.
    predict(model,[row]);row=_rows([row],len(model['feature_names']))[0]
    n=len(model['trees']);prefix=n if prefix is None else prefix
    if type(prefix) is not int or not 0<=prefix<=n:raise ValueError('prefix must be an integer from zero through fitted trees')
    routes=[dict(round=i+1,**route(t,row)) for i,t in enumerate(model['trees'])]
    leaves=[r['leaf'] for r in routes];rate=model['learning_rate'];initial=model['initial_logit']
    states=[]
    for k in range(n+1):
        z=initial+rate*math.fsum(leaves[:k])
        if not math.isfinite(z):raise ValueError('prefix exceeds finite float range')
        states.append(dict(round=k,logit=z,probability=sigmoid(z),label=int(z>=0)))
    chosen=states[prefix]
    return dict(version=1,row=row,prefix=prefix,fitted_rounds=n,initial_logit=initial,learning_rate=rate,states=states,trees=routes,chosen=chosen,scope='frozen additive logit prefix, not feature importance, causal effects, calibrated probability or a new model selection')
