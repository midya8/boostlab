#!/usr/bin/env bash
set -euo pipefail
python -m boostlab demo
python -m boostlab train --input examples/train.json --rounds 6 --min-leaf 2 --patience 6 >/tmp/boostlab-trained.json
python - <<'PY'
import json
from boostlab import predict
with open('/tmp/boostlab-trained.json') as source:report=json.load(source)
rows=[[1],[10]]
assert predict(report['model'],rows)['probabilities'][0]<.5
assert predict(report['model'],rows)['probabilities'][1]>.5
with open('/tmp/boostlab-predict.json','w') as target:
    json.dump({'model':report['model'],'rows':rows},target)
print('serialized training artifact predicts both sides of the boundary')
PY
python -m boostlab predict --input /tmp/boostlab-predict.json
python - <<'PY'
from boostlab import fit
from boostlab.demo import fixture
train,val,_=fixture()
val={'rows':train['rows'],'labels':[1-y for y in train['labels']]}
result=fit({'feature_names':['a','b'],'train':train,'validation':val},rounds=10,patience=3)
assert result['selection']['best_round']==0
assert result['model']['trees']==[]
print('opposite-label validation rejects fitted trees and retains the initial model')
PY
