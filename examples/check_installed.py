"""Check the installed tests and full CLI outputs outside the checkout.

Use --section tests or --section commands to run either half on its own.
The default runs both. These are same-runtime packaging comparisons, not
independent numerical or statistical evidence.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[1]
ENV = os.environ.copy()
ENV.pop('PYTHONPATH', None)
ENV.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
MODULES = ['boostlab', 'boostlab.__main__', 'boostlab.cli', 'boostlab.core',
           'boostlab.demo', 'boostlab.path', 'boostlab.path_export', 'boostlab.slices']
COMMANDS = [
    ['demo'], ['path-report'], ['path', '--input', 'path.json'],
    ['predict', '--input', 'predict.json'],
    ['train', '--input', 'examples/train.json', '--rounds', '6', '--min-leaf', '2'],
    ['train', '--input', 'examples/train.json', '--rounds', '6', '--min-leaf', '2',
     '--leaf-mode', 'newton', '--l2', '1', '--max-leaf-step', '3'],
]


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=True)
    data = json.loads((ROOT / 'boostlab/artifacts/path-depth_1.json').read_text())
    model = data['report']['last_fitted_model']
    (directory / 'path.json').write_text(json.dumps(dict(model=model, row=[0., 0.], prefix=0)))
    (directory / 'predict.json').write_text(json.dumps(dict(model=model, rows=[[0., 0.], [.1, -.1]])))
    shutil.copytree(ROOT / 'examples', directory / 'examples')


def products(directory):
    return {str(path.relative_to(directory)): path.read_bytes()
            for folder in ['boostlab/artifacts', 'docs']
            for path in (directory / folder).glob('*') if path.is_file()}


def call(args, cwd, env=ENV):
    return subprocess.check_output(args, cwd=cwd, env=env, text=True)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--section', choices=['all', 'tests', 'commands'], default='all')
args = parser.parse_args()
with tempfile.TemporaryDirectory() as directory:
    work = Path(directory)
    wheels = work / 'wheels'
    wheels.mkdir()
    call([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '-w', str(wheels), str(ROOT)], work)
    built = list(wheels.glob('*.whl'))
    assert len(built) == 1, built
    venv.create(work / 'env', with_pip=True)
    py = str(work / 'env/bin/python')
    call([py, '-m', 'pip', 'install', str(built[0])], work)
    outside = work / 'outside'
    outside.mkdir()
    for name in ['tests', 'examples', 'docs']:
        shutil.copytree(ROOT / name, outside / name)
    suite = '''
import importlib,importlib.util,json,pathlib,sys,unittest
outside=pathlib.Path.cwd().resolve()
sys.path.insert(0,str(outside))
names=json.loads(sys.argv[1])
source=pathlib.Path(sys.argv[2]).resolve()
prefix=pathlib.Path(sys.prefix).resolve()
for name in names:importlib.import_module(name)
def check():
    for name in names:
        origin=pathlib.Path(importlib.util.find_spec(name).origin).resolve()
        assert origin.is_relative_to(prefix),(name,str(origin))
        assert not origin.is_relative_to(source),(name,str(origin))
        assert 'site-packages' in origin.parts,(name,str(origin))
check()
if sys.argv[3]=='tests':
    tests=unittest.defaultTestLoader.discover(str(outside/'tests'))
    result=unittest.TextTestRunner(verbosity=2).run(tests)
    assert result.testsRun>0 and result.wasSuccessful()
    print(json.dumps(dict(tests=result.testsRun)))
check()
print(json.dumps(dict(installed_origins=len(names),checked_before_and_after=True)))
'''
    if args.section in ['all', 'tests']:
        print(call([py, '-I', '-c', suite, json.dumps(MODULES), str(ROOT), 'tests'], outside))
    if args.section in ['all', 'commands']:
        print(call([py, '-I', '-c', suite, json.dumps(MODULES), str(ROOT), 'imports'], outside))
        reference = work / 'reference'
        prepare(reference)
        for name in ['path.json', 'predict.json']:
            shutil.copyfile(reference / name, outside / name)
        source_env = ENV.copy()
        source_env['PYTHONPATH'] = str(ROOT)
        expected = []
        for command in COMMANDS:
            output = call([sys.executable, '-m', 'boostlab'] + command, reference, source_env).strip()
            expected.append(output)
        export_root = reference / 'export'
        (export_root / 'boostlab').mkdir(parents=True)
        call([sys.executable, '-c', 'from boostlab.path_export import export; export(".")'], export_root, source_env)
        exports = products(export_root)
        assert len(exports) == 23, len(exports)
        for name, data in exports.items():
            assert data == (ROOT / name).read_bytes(), name
        # This package has no console-script entrypoint. Check its module CLI.
        for command, report in zip(COMMANDS, expected):
            output = call([py, '-I', '-m', 'boostlab'] + command, outside).strip()
            assert output == report, command
        exported = outside / 'export'
        (exported / 'boostlab').mkdir(parents=True)
        call([py, '-I', '-c', 'from boostlab.path_export import export; export(".")'], exported)
        assert products(exported) == exports
        print(call([py, '-I', '-c', suite, json.dumps(MODULES), str(ROOT), 'imports'], outside))
        print(json.dumps(dict(command_invocations=len(COMMANDS), command_kinds=5,
                              complete_output_files=len(exports), fresh_full_budget_fits=3,
                              module_entrypoints=1, exporter_api=True)))
