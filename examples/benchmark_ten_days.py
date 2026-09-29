"""Compare six 10-day experiments, serial versus four real worker processes.

Run from the repository: python examples/benchmark_ten_days.py --output-dir
results/ten-day-benchmark. The destination must not already exist.
"""
import argparse
from functools import partial
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

# Set before importing numerical libraries, including in Windows workers.
os.environ['MPLBACKEND'] = 'Agg'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from verify_batch_consistency import build, check_frames, engine, read_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    configs = [dict(id=f'{age}-{number:03d}', patient=f'{age}#{number:03d}',
                    scenario='random', controller='bb', hours=240,
                    seeds=[100 + i, 200 + i, 300 + i])
               for i, (age, number) in enumerate(
                   (age, number) for age in ('child', 'adolescent', 'adult')
                   for number in (1, 2))]
    report = dict(
        python=platform.python_version(), platform=platform.platform(),
        cpu_count=os.cpu_count(), parent_pid=os.getpid(), workers=4,
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       text=True).strip(),
        engine_sha256=sha256(Path(engine.__file__).read_bytes()).hexdigest(),
        harness_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        packages={name: version(name) for name in (
            'numpy', 'scipy', 'pandas', 'pathos', 'multiprocess', 'dill')},
        thread_limits={name: os.environ[name] for name in (
            'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        configs=configs, rounds=2, comparisons=0, batches=[], errors=[],
        patient_comparisons=[], timing_scope=(
            'Wall time of batch_sim: simulation, CSV writes, pool startup, '
            'result transfer and pool cleanup; excludes object construction, '
            'imports and result comparison. Sequential batches, no overlap.'))
    destination = root / 'report.json'

    def checkpoint():
        destination.write_text(json.dumps(report, indent=2), encoding='utf-8')

    original_pool = engine.Pool
    engine.Pool = partial(original_pool, nodes=4)
    reference = {}
    try:
        if not engine.pathos:
            raise RuntimeError('Real pathos multiprocessing is required')
        for repeat, order in enumerate(((False, True), (True, False)), 1):
            frames_by_mode, saved_by_mode, files_by_mode = {}, {}, {}
            for parallel in order:
                mode = 'parallel' if parallel else 'serial'
                directory = root / f'round-{repeat}' / mode
                tasks = [build(c, directory) for c in configs]
                print(f'START round={repeat} mode={mode} patients=6 days=10',
                      flush=True)
                started = time.perf_counter()
                frames = engine.batch_sim(tasks, parallel=parallel)
                elapsed = time.perf_counter() - started
                pids = sorted({frame.attrs['worker_pid'] for frame in frames})
                assert len(frames) == len(configs)
                assert ((len(pids) == 4 and os.getpid() not in pids)
                        if parallel else pids == [os.getpid()])
                files = [directory / (c['patient'] + '.csv') for c in configs]
                assert len(list(directory.glob('*.csv'))) == len(configs)
                saved = [read_csv(file) for file in files]
                details = []
                for config, frame, disk in zip(configs, frames, saved):
                    label = f'round-{repeat}/{mode}/{config["patient"]}'
                    check_frames(frame, disk, label + '/memory-vs-csv', report)
                    # Dexcom samples every three minutes; initial row included.
                    assert len(frame) == 4801 and len(frame.columns) == 7
                    assert str(frame.index[0]) == '2018-01-01 00:00:00'
                    assert str(frame.index[-1]) == '2018-01-11 00:00:00'
                    daily_cho = frame['CHO'].iloc[:-1].resample('D').sum()
                    assert len(daily_cho) == 10 and (daily_cho > 0).all()
                    details.append(dict(patient=config['patient'],
                                        rows=len(frame),
                                        start=str(frame.index[0]),
                                        end=str(frame.index[-1]),
                                        days_with_meals=int((daily_cho > 0).sum())))
                    if mode in reference:
                        idx = configs.index(config)
                        check_frames(reference[mode][idx], frame,
                                     label + '/repeatability', report)
                report['batches'].append(dict(
                    round=repeat, mode=mode, seconds=elapsed, pids=pids,
                    patients=details))
                frames_by_mode[mode], saved_by_mode[mode] = frames, saved
                files_by_mode[mode] = files
                reference.setdefault(mode, frames)
                checkpoint()
                print(f'DONE round={repeat} mode={mode} seconds={elapsed:.3f} '
                      f'pids={pids}', flush=True)
            for i, config in enumerate(configs):
                left, right = (frames_by_mode[m][i]
                               for m in ('serial', 'parallel'))
                label = f'round-{repeat}/{config["patient"]}'
                check_frames(left, right, label + '/serial-vs-parallel', report)
                check_frames(saved_by_mode['serial'][i],
                             saved_by_mode['parallel'][i],
                             label + '/csv-vs-csv', report)
                hashes = {m: sha256(files_by_mode[m][i].read_bytes()).hexdigest()
                          for m in ('serial', 'parallel')}
                assert hashes['serial'] == hashes['parallel']
                report['patient_comparisons'].append(dict(
                    round=repeat, patient=config['patient'],
                    max_absolute_error_by_column=(left-right).abs().max().to_dict(),
                    csv_sha256=hashes, csv_bytes_identical=True))
            checkpoint()
        report['timing'] = []
        for repeat in (1, 2):
            seconds = {b['mode']: b['seconds'] for b in report['batches']
                       if b['round'] == repeat}
            report['timing'].append(dict(
                round=repeat, **seconds,
                speedup=seconds['serial'] / seconds['parallel']))
        totals = {mode: sum(b['seconds'] for b in report['batches']
                            if b['mode'] == mode)
                  for mode in ('serial', 'parallel')}
        report['aggregate_speedup'] = totals['serial'] / totals['parallel']
        report['mean_seconds'] = {mode: total/2 for mode, total in totals.items()}
    except Exception:
        report['errors'].append(dict(check='execution',
                                     message=traceback.format_exc()))
    finally:
        engine.Pool = original_pool
    report['passed'] = not report['errors']
    checkpoint()
    print(f'FINAL passed={report["passed"]} comparisons={report["comparisons"]} '
          f'report={destination}', flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
