# batch_sim silently overwrites same-patient scenarios and can race when creating the output directory

## Summary

At commit `a6f6777586c860457a2a1e6cce6d0bd3fd33d1cc` (package version
0.2.11), two `SimObj` instances for the same patient and output directory
return two distinct DataFrames, but save only one CSV. This happens with both
`parallel=False` and `parallel=True`. In serial execution, the second task
overwrites the first. In process execution, the filename is shared between
workers, so there is no task-specific destination.

This is a persistence problem: the seeded simulation values were exactly equal
between serial and process execution in the tested matrix. It should not be
described as a demonstrated numerical solver nondeterminism problem.

## Environment

- Windows 10, build 19045; Python 3.11.15.
- NumPy 1.26.4, SciPy 1.11.4, pandas 2.3.3.
- pathos 0.3.5, multiprocess 0.70.19, dill 0.4.1.
- matplotlib 3.8.4 with `MPLBACKEND=Agg`.
- Detailed dependency snapshot: `docs/evidence/batch-consistency/requirements.txt`.

## Minimal reproduction

Save this as `reproduce.py` in a checkout with `pip install -e .` installed:

To use the tested numerical/process stack with Python 3.11:

```shell
python -m pip install -e . "numpy==1.26.4" "scipy==1.11.4" "pandas==2.3.3" "matplotlib==3.8.4" "pathos==0.3.5" "multiprocess==0.70.19" "dill==0.4.1" "setuptools<81"
```

```python
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from simglucose.actuator.pump import InsulinPump
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation import sim_engine as engine


def task(path, meal):
    env = T1DSimEnv(
        T1DPatient.withName('adolescent#001', random_init_bg=True, seed=1),
        CGMSensor.withName('Dexcom', seed=2),
        InsulinPump.withName('Insulet'),
        CustomScenario(datetime(2018, 1, 1), [(1, meal)]),
    )
    return engine.SimObj(env, BBController(), timedelta(hours=3),
                         animate=False, path=path)


if __name__ == '__main__':
    # Bound resources while still exercising the real process backend.
    engine.Pool = partial(engine.Pool, nodes=2)
    for parallel in (False, True):
        with TemporaryDirectory() as directory:
            frames = engine.batch_sim(
                [task(directory, 20), task(directory, 50)], parallel=parallel)
            files = sorted(p.name for p in Path(directory).glob('*.csv'))
            print('parallel:', parallel,
                  'different trajectories:', not frames[0].equals(frames[1]),
                  'saved:', files)
```

Run `python reproduce.py`.

**Actual on the baseline:** both modes report different trajectories, but the
saved file list contains only `adolescent#001.csv`.

**Expected:** both experiment results should remain available, with a stable
mapping from input tasks to output files in either execution mode.

The repository's `examples/advanced_tutorial.py` constructs the same
patient twice for different scenarios using the same directory, so this is
also a use case shown by the project itself.

## Related directory creation race

`SimObj.save_results()` currently performs:

```python
if not os.path.isdir(self.path):
    os.makedirs(self.path)
```

If two workers observe a missing directory before either creates it, one can
raise `FileExistsError`. A regression test forces precisely this interleaving
using actual filesystem operations and fails on the baseline with WinError 183.
Ordinary process scheduling does not reproduce the race on every run; the
deterministic interleaving test and the uncontrolled integration runs are
reported separately.

## Proposed fix

- Allocate filenames in the parent before dispatch, using normalized output
  paths and input order. Preserve `patient.csv` for unique destinations. Assign
  `patient__N.csv` for duplicate destinations, reserving other tasks' original
  names so generated suffixes cannot collide with them.
- Use `os.makedirs(path, exist_ok=True)`; do not suppress unrelated I/O errors.
- Keep the returned DataFrame order and model equations unchanged.

Repeating the same batch in the same directory intentionally updates the same
files. Separate concurrently running batches should use separate run
directories; the proposed fix does not introduce a cross-program file lock.

## Verification artifacts

- Regression tests: `tests/test_batch_consistency.py`.
- Independent harness: `python examples/verify_batch_consistency.py
  --output-dir results/new-check` (the directory must not already exist).
- Tested matrix: one patient per age group, custom/random meals, BB/PID,
  two seed triplets, 24 simulated hours, three repetitions, two actual workers.
- Before/after reports and exact commands: `docs/serial-parallel-validation.md`.

This is a draft for review; it has not been posted to GitHub.
