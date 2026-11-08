"""Freeze existing demo protocol; complete models, histories and native references."""
import json,math
from pathlib import Path
from .core import fit,predict,logistic_loss
from .demo import fixture,metrics,run_demo
from .path import inspect

CONFIGS=[('depth_1',1,'mean',0),('depth_2',2,'mean',0),('newton_depth_2',2,'newton',1)]

def build():
    train,val,test=fixture();payload=dict(feature_names=['a','b'],train=train,validation=val);arms={}
    for name,depth,mode,l2 in CONFIGS:
        report=fit(payload,rounds=80,depth=depth,learning_rate=.5,min_leaf=5,patience=80,leaf_mode=mode,l2=l2,max_leaf_step=3)
        full=report['last_fitted_model'];history=[]
        for k in range(len(full['trees'])+1):
            m=dict(full,trees=full['trees'][:k]);h=dict(round=k)
            for split,data in [('train',train),('validation',val),('test',test)]:h[split]=metrics(m,data)
            history.append(h)
        refs=[inspect(full,row) for row in test['rows'][:20]]
        arms[name]=dict(report=report,history=history,references=refs)
    edited_rows=[[0.,0.],[-2.,2.],[2.,-2.],[.125,.875],[1e6,-1e6],[-1e6,1e6]]
    edited={name:[inspect(arm['report']['last_fitted_model'],row) for row in edited_rows] for name,arm in arms.items()}
    return dict(edited_references=edited,version=1,protocol=dict(seed=7,rounds=80,rate=.5,min_leaf=5,patience=80,display_test_indices=list(range(20)),selection='original earliest validation logloss improvement >1e-12; test metrics are report-only'),samples=dict(rows=test['rows'][:20],labels=test['labels'][:20]),arms=arms,original_demo=run_demo(),scope='existing inspected synthetic sign-of-product development fixture with 5% flipped labels; not external or untouched benchmark, causal isolation, confidence or calibrated probabilities')

def encoded(v):return json.dumps(v,separators=(',',':'),allow_nan=False)+'\n'

def export(root):
    out=build();root=Path(root);(root/'boostlab/artifacts').mkdir(exist_ok=True);(root/'docs').mkdir(exist_ok=True)
    # One bounded file per arm; all20complete path witnesses and every prefix retained.
    for name,arm in out['arms'].items():
        stored={k:v for k,v in arm.items() if k!='references'}
        (root/('boostlab/artifacts/path-'+name+'.json')).write_text(encoded(stored))
        for chunk in range(4):(root/('boostlab/artifacts/path-'+name+'-refs-'+str(chunk)+'.json')).write_text(encoded(arm['references'][chunk*5:(chunk+1)*5]))
        browser=dict(arm);browser.pop('references');browser['model']=browser['report']['last_fitted_model'];browser['selection']=browser['report']['selection'];browser.pop('report')
        (root/('docs/path-'+name+'.js')).write_text('globalThis.BOOST_ARMS ??= {};globalThis.BOOST_ARMS['+json.dumps(name)+'] = '+encoded(browser).rstrip()+';\n')
    meta={k:v for k,v in out.items() if k not in ['arms','edited_references']}
    for name,refs in out['edited_references'].items():(root/('boostlab/artifacts/path-'+name+'-edits.json')).write_text(encoded(refs))
    (root/'boostlab/artifacts/path-meta.json').write_text(encoded(meta));browser_meta={k:v for k,v in meta.items() if k!='edited_references'}
    (root/'docs/data.js').write_text('globalThis.BOOST_DATA = '+encoded(browser_meta).rstrip()+';\n')
    return dict(arms=3,prefix_states=243,sample_prefixes=4860,full_paths=60)

def saved():
    root=Path(__file__).parent/'artifacts';out=json.loads((root/'path-meta.json').read_text());out['arms']={name:json.loads((root/('path-'+name+'.json')).read_text()) for name,_,_,_ in CONFIGS}
    out['edited_references']={name:json.loads((root/('path-'+name+'-edits.json')).read_text()) for name in out['arms']}
    for name in out['arms']:out['arms'][name]['references']=[v for i in range(4) for v in json.loads((root/('path-'+name+'-refs-'+str(i)+'.json')).read_text())]
    return out

if __name__=='__main__':
    print(json.dumps(export(Path(__file__).resolve().parents[1])))
