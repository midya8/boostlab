import argparse
import json
import sys
from .core import fit,predict
from .demo import run_demo


def _pairs(items):
    out={}
    for k,v in items:
        if k in out:raise ValueError("duplicate JSON key: "+k)
        out[k]=v
    return out

def main(argv=None):
    parser=argparse.ArgumentParser(prog='boostlab');sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('demo');t=sub.add_parser('train');t.add_argument('--input',required=True)
    t.add_argument('--rounds',type=int,default=80);t.add_argument('--depth',type=int,default=2)
    t.add_argument('--learning-rate',type=float,default=.3);t.add_argument('--min-leaf',type=int,default=5)
    t.add_argument('--patience',type=int,default=10)
    t.add_argument('--leaf-mode',choices=['mean','newton'],default='mean')
    t.add_argument('--l2',type=float,default=0.);t.add_argument('--max-leaf-step',type=float,default=3.)
    p=sub.add_parser('predict');p.add_argument('--input',required=True)
    p=sub.add_parser('path');p.add_argument('--input',required=True)
    sub.add_parser('path-report')
    args=parser.parse_args(argv)
    try:
        if args.command=='demo':result=run_demo()
        elif args.command=='path-report':
            from .path_export import saved
            r=saved();result=dict(protocol=r['protocol'],selection={k:v['report']['selection'] for k,v in r['arms'].items()},original_demo=r['original_demo'],scope=r['scope'])
        else:
            if args.input=='-':payload=json.load(sys.stdin,object_pairs_hook=_pairs)
            else:
                with open(args.input,encoding='utf-8') as source:payload=json.load(source,object_pairs_hook=_pairs)
            if args.command=='path':
                from .path import inspect
                if not isinstance(payload,dict) or set(payload)!={'model','row','prefix'}:raise ValueError('path input needs only model,row,prefix')
                result=inspect(payload['model'],payload['row'],payload['prefix'])
            elif args.command=='train':result=fit(payload,args.rounds,args.depth,args.learning_rate,args.min_leaf,args.patience,
                                                args.leaf_mode,args.l2,args.max_leaf_step)
            else:
                if not isinstance(payload,dict) or set(payload)!={'model','rows'}:
                    raise ValueError('predict input requires model and rows only')
                result=predict(payload['model'],payload['rows'])
        body=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)
    except (ValueError,TypeError,OSError,OverflowError) as exc:
        print('boostlab input error: '+str(exc),file=sys.stderr);return 2
    print(body);return 0
