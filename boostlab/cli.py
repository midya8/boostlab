import argparse
import json
import sys
from .core import fit,predict
from .demo import run_demo


def main(argv=None):
    parser=argparse.ArgumentParser(prog='boostlab');sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('demo');t=sub.add_parser('train');t.add_argument('--input',required=True)
    t.add_argument('--rounds',type=int,default=80);t.add_argument('--depth',type=int,default=2)
    t.add_argument('--learning-rate',type=float,default=.3);t.add_argument('--min-leaf',type=int,default=5)
    t.add_argument('--patience',type=int,default=10)
    p=sub.add_parser('predict');p.add_argument('--input',required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='demo':result=run_demo()
        else:
            if args.input=='-':payload=json.load(sys.stdin)
            else:
                with open(args.input,encoding='utf-8') as source:payload=json.load(source)
            if args.command=='train':result=fit(payload,args.rounds,args.depth,args.learning_rate,args.min_leaf,args.patience)
            else:
                if not isinstance(payload,dict) or set(payload)!={'model','rows'}:
                    raise ValueError('predict input requires model and rows only')
                result=predict(payload['model'],payload['rows'])
        body=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)
    except (ValueError,TypeError,OSError,OverflowError) as exc:
        print('boostlab input error: '+str(exc),file=sys.stderr);return 2
    print(body);return 0
