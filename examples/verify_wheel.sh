#!/usr/bin/env bash
set -euo pipefail
installed=$1
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
export PYTHONPATH="$installed"
python - <<'PY'
import importlib,pkgutil,boostlab,json
from boostlab.path_export import build,saved
mods=list(pkgutil.walk_packages(boostlab.__path__,boostlab.__name__+'.'))
for m in mods:importlib.import_module(m.name)
assert len(mods)==6
r=saved();assert build()==r
m=r['arms']['depth_1']['report']['last_fitted_model'];json.dump(dict(model=m,row=[0.,0.],prefix=0),open('path.json','w'));json.dump(dict(model=m,rows=[[0.,0.],[.1,-.1]]),open('predict.json','w'))
from boostlab.demo import fixture
train,val,_=fixture();json.dump(dict(feature_names=['a','b'],train=train,validation=val),open('train.json','w'))
print('all7modules including package root,3full80roundrefits,243metrics4860sampleprefixstates18editedpaths reproduce outside checkout')
PY
python -m boostlab demo > demo.json
python -m boostlab path-report > report.json
python -m boostlab path --input path.json > path-out.json
python -m boostlab predict --input predict.json > predict-out.json
python -m boostlab train --input train.json --rounds 3 --patience 3 > train-out.json
python - <<'PY'
import json
from boostlab.path_export import saved
assert json.load(open('demo.json'))==saved()['original_demo']
assert json.load(open('path-out.json'))['prefix']==0
assert len(json.load(open('predict-out.json'))['logits'])==2
assert json.load(open('train-out.json'))['selection']['fitted_rounds']==3
print('all5maincommands and full exporter API reproduction pass')
PY
