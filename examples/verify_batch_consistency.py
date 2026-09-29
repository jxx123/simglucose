"""Reproduce batch output loss and verify seeded serial/process equivalence.

Run after ``pip install -e .``. This harness bounds pathos to two real worker
processes; it does not replace the simulator, random generators, or solver.
Every invocation needs a new output directory so old CSVs cannot mask failures.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback
from uuid import uuid4

os.environ.setdefault('MPLBACKEND', 'Agg')

import pandas as pd
from pandas.testing import assert_frame_equal
from pathos.multiprocessing import ProcessPool

from simglucose.actuator.pump import InsulinPump
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.controller.pid_ctrller import PIDController
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation import sim_engine as engine


class ObservedSimulation(engine.SimObj):
    def results(self):
        frame = super().results()
        frame.attrs['worker_pid'] = os.getpid()
        return frame


def build(config, directory):
    start = datetime(2018, 1, 1)
    patient_seed, sensor_seed, scenario_seed = config['seeds']
    scenario = (RandomScenario(start, seed=scenario_seed)
                if config['scenario'] == 'random' else
                CustomScenario(start, config['meals']))
    env = T1DSimEnv(
        T1DPatient.withName(config['patient'], random_init_bg=True,
                           seed=patient_seed),
        CGMSensor.withName('Dexcom', seed=sensor_seed),
        InsulinPump.withName('Insulet'), scenario)
    controller = (BBController() if config['controller'] == 'bb' else
                  PIDController(P=0.001, I=0.00001, D=0.001, target=140))
    return ObservedSimulation(env, controller, timedelta(hours=config['hours']),
                              animate=False, path=directory)


def read_csv(path):
    return pd.read_csv(path, index_col='Time', parse_dates=['Time'],
                       float_precision='round_trip')


def check_frames(expected, actual, label, report):
    report['comparisons'] += 1
    try:
        assert_frame_equal(expected, actual, check_exact=True)
    except AssertionError as exc:
        detail = {'check': label, 'message': str(exc)}
        if (expected.index.equals(actual.index)
                and expected.columns.equals(actual.columns)):
            unequal = ~(expected.eq(actual) | (expected.isna() & actual.isna()))
            for column in expected.columns:
                rows = unequal.index[unequal[column]]
                if len(rows):
                    detail['first_difference'] = {
                        'column': column, 'time': str(rows[0]),
                        'expected': float(expected.at[rows[0], column]),
                        'actual': float(actual.at[rows[0], column])}
                    break
            absolute = (expected - actual).abs()
            relative = absolute / expected.abs().replace(0, float('nan'))
            detail['max_absolute_error'] = float(absolute.max().max())
            detail['max_relative_error_nonzero_reference'] = float(
                relative.max().max())
        report['errors'].append(detail)


def run_batch(tasks, parallel, label, report):
    started = time.perf_counter()
    frames = engine.batch_sim(tasks, parallel=parallel)
    if len(frames) != len(tasks):
        raise AssertionError(f'{label}: {len(frames)} results for {len(tasks)} tasks')
    pids = sorted({frame.attrs['worker_pid'] for frame in frames})
    report['batches'].append({
        'label': label, 'parallel': parallel, 'pids': pids,
        'tasks': len(tasks), 'seconds': time.perf_counter() - started})
    if parallel and (os.getpid() in pids or len(pids) < 2):
        report['errors'].append({'check': label,
                                 'message': 'Two worker PIDs not observed'})
    return frames


def matrix(root, repeats, report):
    configs = []
    for age in ('adolescent', 'adult', 'child'):
        for scenario in ('custom', 'random'):
            for controller in ('bb', 'pid'):
                for seeds in ((1, 2, 3), (11, 12, 13)):
                    configs.append(dict(
                        id=f'{age}-{scenario}-{controller}-{seeds[0]}',
                        patient=age + '#001', scenario=scenario,
                        controller=controller, seeds=seeds, hours=24,
                        meals=[(7, 45), (12, 70), (18, 80)]))
    report['matrix_configs'] = configs
    reference = None
    for repeat in range(repeats):
        modes = {}
        for parallel in (False, True):
            mode = 'parallel' if parallel else 'serial'
            directories = [root / str(repeat) / mode / c['id'] for c in configs]
            tasks = [build(c, d) for c, d in zip(configs, directories)]
            frames = run_batch(tasks, parallel, f'matrix/{repeat}/{mode}', report)
            for c, d, frame in zip(configs, directories, frames):
                file = d / (c['patient'] + '.csv')
                check_frames(frame, read_csv(file), str(file), report)
            modes[mode] = frames
        for c, left, right in zip(configs, modes['serial'], modes['parallel']):
            check_frames(left, right, f'matrix/{repeat}/{c["id"]}', report)
        if reference is not None:
            for c, before, after in zip(configs, reference, modes['serial']):
                check_frames(before, after, f'repeat/{repeat}/{c["id"]}', report)
        reference = modes['serial']


def duplicates(root, report):
    base = dict(patient='adolescent#001', scenario='custom', controller='bb',
                seeds=(1, 2, 3), hours=3)
    configs = [dict(base, meals=[(1, 20), (2, 15)]),
               dict(base, meals=[(1, 50)])]
    modes = {}
    saved_modes = {}
    for parallel in (False, True):
        mode = 'parallel' if parallel else 'serial'
        directory = root / mode
        # Pre-create here to isolate overwrites from directory creation races.
        directory.mkdir(parents=True)
        frames = run_batch([build(c, directory) for c in configs], parallel,
                           'duplicates/' + mode, report)
        files = sorted(directory.glob('*.csv'))
        saved = {file.name: read_csv(file) for file in files}
        matches = [sum(frame.equals(item) for item in saved.values())
                   for frame in frames]
        report['duplicates'][mode] = {'files': list(saved), 'matches': matches}
        if len(saved) != 2 or matches != [1, 1]:
            report['errors'].append({'check': 'duplicates/' + mode,
                                     'message': 'Both scenarios must be saved',
                                     'files': list(saved), 'matches': matches})
        modes[mode] = frames
        saved_modes[mode] = saved
    for index, (left, right) in enumerate(zip(modes['serial'], modes['parallel'])):
        check_frames(left, right, f'duplicate-values/{index}', report)
    if saved_modes['serial'].keys() != saved_modes['parallel'].keys():
        report['errors'].append({'check': 'duplicate-files',
                                 'message': 'Mode-dependent output filenames'})
    else:
        for name in saved_modes['serial']:
            check_frames(saved_modes['serial'][name],
                         saved_modes['parallel'][name], name, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--case', choices=('all', 'matrix', 'duplicates'),
                        default='all')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(python=platform.python_version(), platform=platform.platform(),
                  cpu_count=os.cpu_count(), parent_pid=os.getpid(), workers=2,
                  case=args.case, repeats=args.repeats,
                  packages={name: version(name) for name in (
                      'numpy', 'scipy', 'pandas', 'pathos', 'multiprocess',
                      'dill', 'gym', 'gymnasium', 'matplotlib', 'setuptools')},
                  engine_sha256=sha256(Path(engine.__file__).read_bytes()).hexdigest(),
                  comparisons=0, batches=[], duplicates={}, errors=[])
    report['commit'] = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip()
    original_pool = engine.Pool

    @contextmanager
    def bounded_pool(**kwargs):
        kwargs.setdefault('id', uuid4().hex)
        pool = ProcessPool(nodes=2, **kwargs)
        try:
            yield pool
        finally:
            # Upstream pathos __exit__ does not clear its cached worker pool.
            # Bound resources even while testing the unmodified simulator.
            pool.clear()

    engine.Pool = bounded_pool
    try:
        if not engine.pathos:
            raise RuntimeError('pathos unavailable; refusing serial fallback')
        if args.case in ('all', 'matrix'):
            matrix(args.output_dir / 'matrix', args.repeats, report)
        if args.case in ('all', 'duplicates'):
            duplicates(args.output_dir / 'duplicates', report)
    except Exception:
        report['errors'].append({'check': 'execution',
                                 'message': traceback.format_exc()})
    finally:
        engine.Pool = original_pool
    report['passed'] = not report['errors']
    destination = args.output_dir / 'report.json'
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Report: {destination}; passed={report["passed"]}; '
          f'comparisons={report["comparisons"]}; errors={len(report["errors"])}')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
