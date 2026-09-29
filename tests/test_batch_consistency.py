"""Regression tests for actual process execution and complete batch outputs."""
from datetime import datetime, timedelta
import os
from pathlib import Path
from uuid import uuid4

import pandas as pd
from pandas.testing import assert_frame_equal
from pathos.multiprocessing import ProcessPool
import pytest

from simglucose.actuator.pump import InsulinPump
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.controller.pid_ctrller import PIDController
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation import sim_engine as engine


class ObservedSimulation(engine.SimObj):
    """Record execution identity without modifying simulated observations."""

    def results(self):
        frame = super().results()
        frame.attrs['worker_pid'] = os.getpid()
        return frame


class MemorySimulation(ObservedSimulation):
    saved = False

    def save_results(self):
        self.saved = True

    def results(self):
        frame = super().results()
        frame.attrs['custom_writer_called'] = self.saved
        return frame


def make_sim(path, patient='adolescent#001', meals=((1, 20), (2, 15)),
             pid=False, simulation_class=ObservedSimulation):
    env = T1DSimEnv(
        T1DPatient.withName(patient, random_init_bg=True, seed=1),
        CGMSensor.withName('Dexcom', seed=2),
        InsulinPump.withName('Insulet'),
        CustomScenario(datetime(2018, 1, 1), list(meals)),
    )
    controller = (PIDController(P=0.001, I=0.00001, D=0.001, target=140)
                  if pid else BBController())
    return simulation_class(env, controller, timedelta(hours=3),
                            animate=False, path=path)


@pytest.fixture
def two_workers(monkeypatch):
    """Bound resources, but exercise the real Windows-spawn pathos backend."""
    pools = []

    def pool(**kwargs):
        kwargs.setdefault('id', uuid4().hex)
        instance = ProcessPool(nodes=2, **kwargs)
        pools.append(instance)
        return instance

    monkeypatch.setattr(engine, 'Pool', pool)
    assert engine.pathos, 'A serial fallback is not a parallel test'
    yield
    for instance in pools:
        instance.clear()


def read_result(path):
    return pd.read_csv(path, index_col='Time', parse_dates=['Time'],
                       float_precision='round_trip')


@pytest.mark.parametrize('parallel', [False, True])
def test_same_patient_different_scenarios_are_both_saved(
        tmp_path, two_workers, parallel):
    tasks = [make_sim(tmp_path), make_sim(tmp_path, meals=((1, 50),))]
    frames = engine.batch_sim(tasks, parallel=parallel)
    assert not frames[0].equals(frames[1]), 'Scenarios must be distinguishable'
    files = sorted(tmp_path.glob('*.csv'))
    assert len(files) == 2, 'A batch must not overwrite another scenario'
    saved = [read_result(path) for path in files]
    for frame in frames:
        assert sum(frame.equals(other) for other in saved) == 1


def test_save_results_accepts_directory_created_by_another_worker(
        tmp_path, monkeypatch):
    directory = tmp_path / 'new'
    task = make_sim(directory)
    task.simulate()
    original_makedirs = os.makedirs

    def another_worker_creates_first(path, *args, **kwargs):
        # A deterministic interleaving of the real filesystem operations:
        # the other worker wins between this worker's check and mkdir.
        if Path(path) == directory and not directory.exists():
            original_makedirs(path)
        return original_makedirs(path, *args, **kwargs)

    monkeypatch.setattr(engine.os, 'makedirs', another_worker_creates_first)
    task.save_results()
    assert_frame_equal(read_result(directory / 'adolescent#001.csv'),
                       task.results(), check_exact=True)


def test_alias_paths_and_generated_names_do_not_collide(tmp_path, two_workers):
    # Two spellings of one output directory must be grouped together. The
    # natural filename of a third patient must not be stolen by a suffix.
    alias = tmp_path / 'subdirectory' / '..'
    tasks = [make_sim(tmp_path), make_sim(alias, meals=((1, 50),)),
             make_sim(tmp_path, meals=((1, 70),))]
    tasks[2].env.patient._params['Name'] = 'adolescent#001__1'
    tasks[2].env.patient.name = 'adolescent#001__1'
    frames = engine.batch_sim(tasks, parallel=True)
    files = list(tmp_path.glob('*.csv'))
    assert len(files) == 3
    assert (tmp_path / 'adolescent#001__1.csv').exists()
    saved = [read_result(path) for path in files]
    for frame in frames:
        assert sum(frame.equals(other) for other in saved) == 1


def test_distinct_directories_keep_legacy_patient_filenames(tmp_path, two_workers):
    directories = [tmp_path / 'one', tmp_path / 'two']
    frames = engine.batch_sim((make_sim(path) for path in directories),
                              parallel=True)
    for directory, frame in zip(directories, frames):
        files = list(directory.glob('*.csv'))
        assert [file.name for file in files] == ['adolescent#001.csv']
        assert_frame_equal(read_result(files[0]), frame, check_exact=True)


def test_parallel_batch_releases_its_workers(tmp_path, two_workers):
    from multiprocess import active_children

    frames = engine.batch_sim([make_sim(tmp_path / 'one'),
                               make_sim(tmp_path / 'two')], parallel=True)
    worker_pids = {frame.attrs['worker_pid'] for frame in frames}
    assert not worker_pids.intersection(p.pid for p in active_children())


def test_parallel_failure_releases_its_workers(tmp_path, two_workers):
    from multiprocess import active_children

    existing_children = {p.pid for p in active_children()}
    invalid_directory = tmp_path / 'regular-file'
    invalid_directory.write_text('not a directory')
    with pytest.raises(FileExistsError):
        engine.batch_sim([make_sim(invalid_directory),
                          make_sim(tmp_path / 'valid')], parallel=True)
    assert not {p.pid for p in active_children()} - existing_children


@pytest.mark.parametrize('parallel', [False, True])
def test_custom_persistence_without_directory_still_runs(two_workers, parallel):
    frames = engine.batch_sim(
        [make_sim(None, simulation_class=MemorySimulation),
         make_sim(None, simulation_class=MemorySimulation)], parallel=parallel)
    assert len(frames) == 2
    assert all(frame.attrs['custom_writer_called'] for frame in frames)
    assert_frame_equal(frames[0], frames[1], check_exact=True)


def test_repeated_batches_match_in_memory_and_on_disk(tmp_path, two_workers):
    patients = ['adolescent#001', 'adult#001', 'child#001']
    reference = None
    for repeat in range(3):
        modes = {}
        for parallel in (False, True):
            root = tmp_path / str(repeat) / str(parallel)
            tasks = [make_sim(root, patient=name, pid=(i == 1))
                     for i, name in enumerate(patients)]
            frames = engine.batch_sim(tasks, parallel=parallel)
            if parallel:
                pids = {frame.attrs['worker_pid'] for frame in frames}
                assert os.getpid() not in pids
                assert len(pids) == 2
            for name, frame in zip(patients, frames):
                assert_frame_equal(read_result(root / (name + '.csv')),
                                   frame, check_exact=True)
            modes[parallel] = frames
        for serial, parallel in zip(modes[False], modes[True]):
            assert_frame_equal(serial, parallel, check_exact=True)
        if reference is not None:
            for before, after in zip(reference, modes[False]):
                assert_frame_equal(before, after, check_exact=True)
        reference = modes[False]


def test_reset_and_reuse_keep_duplicate_outputs_stable(tmp_path, two_workers):
    saved_by_mode = {}
    reference = None
    for parallel in (False, True):
        root = tmp_path / str(parallel)
        root.mkdir()
        tasks = [make_sim(root), make_sim(root, meals=((1, 50),))]
        previous_files = None
        for _ in range(3):
            for task in tasks:
                task.reset()
            frames = engine.batch_sim(tasks, parallel=parallel)
            if reference is None:
                reference = frames
            for left, right in zip(reference, frames):
                assert_frame_equal(left, right, check_exact=True)
            files = {p.name: p.read_bytes() for p in root.glob('*.csv')}
            assert len(files) == 2
            if previous_files is not None:
                assert files == previous_files
            previous_files = files
        saved_by_mode[parallel] = previous_files
    assert saved_by_mode[False] == saved_by_mode[True]


def test_high_level_ui_produces_identical_reports_and_patient_csvs(
        tmp_path, two_workers):
    import matplotlib.pyplot as plt
    from simglucose.simulation.user_interface import simulate

    outputs = {}
    patients = ['adolescent#001', 'adult#001', 'child#001']
    for parallel in (False, True):
        root = tmp_path / str(parallel)
        stats = simulate(
            sim_time=timedelta(hours=3),
            scenario=CustomScenario(datetime(2018, 1, 1), [(1, 20), (2, 15)]),
            controller=BBController(), patient_names=patients,
            cgm_name='Dexcom', cgm_seed=2, insulin_pump_name='Insulet',
            start_time=datetime(2018, 1, 1), save_path=root,
            animate=False, parallel=parallel)
        outputs[parallel] = (stats, {name: read_result(root / (name + '.csv'))
                                    for name in patients})
        plt.close('all')
    assert_frame_equal(outputs[False][0], outputs[True][0], check_exact=True)
    for name in patients:
        assert_frame_equal(outputs[False][1][name], outputs[True][1][name],
                           check_exact=True)
