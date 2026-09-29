# simglucose serial/process consistency: findings, fixes, and validation

[中文版](serial-process-report.zh-CN.md)

**Report date: 2026-09-30. Status: local implementation and validation complete; no push, issue, or pull request has been published.**

## 1. Summary

This investigation compares serial and process-based batch simulations under identical configurations, explicit seeds, and one runtime environment. It covers returned data, persisted CSVs, and repeated execution.

1. Seeded numerical trajectories already matched in the tested baseline matrix. No numerical solver or random-sequence divergence was reproduced.
2. Confirmed defects were overwritten same-patient CSV outputs, a directory-creation race, and retained pathos worker processes after a batch returned.
3. The final implementation passed **27 repository tests**, **268 exact matrix comparisons**, and **60 exact comparisons** in an additional six-patient, ten-day benchmark.
4. In that benchmark, mean serial time was **174.10 seconds**, versus **65.55 seconds** with four processes: approximately **2.66× speedup** on this machine. This compares two modes of the fixed implementation, not baseline-versus-patch performance.

## 2. Versions, environment, and concurrency boundary

| Item | Recorded value |
| --- | --- |
| Upstream | [jxx123/simglucose](https://github.com/jxx123/simglucose) |
| Contributor fork | [FantasticLYW/simglucose_issue](https://github.com/FantasticLYW/simglucose_issue) |
| Baseline | `a6f6777586c860457a2a1e6cce6d0bd3fd33d1cc`, package version 0.2.11 |
| Implementation commit | `afe5682b03a866d25a37e2ea57961368b0e508dc` |
| Ten-day validation commit | `7e9f3fe` |
| Local branch | `fix/batch-consistency` |
| OS / CPU | Windows 10 build 19045 / AMD Ryzen 7 4800H, 8 cores, 16 logical processors |
| Python | 3.11.15 |
| Numerical libraries | NumPy 1.26.4, SciPy 1.11.4, pandas 2.3.3 |
| Process libraries | pathos 0.3.5, multiprocess 0.70.19, dill 0.4.1 |
| Other main dependencies | matplotlib 3.8.4, gym 0.9.4, gymnasium 0.29.1 |

The complete dependency snapshot is in [requirements.txt](evidence/batch-consistency/requirements.txt).

A `SimObj` represents one experiment with a patient, devices, controller, meal scenario, and duration. `batch_sim(..., parallel=False)` runs experiments sequentially; `parallel=True` distributes separate experiments among processes. Time steps inside each experiment remain sequential. A single experiment is not divided among workers, so putting one experiment into a process pool adds startup and transfer overhead without parallelizing its time steps.

## 3. Reproduced findings and root causes

### 3.1 Different experiments overwrite the same patient's CSV

The baseline derives the output filename only from the patient name. Two scenarios sharing a patient and directory return two different DataFrames but leave only one `adolescent#001.csv`. Serial execution overwrites the earlier file; process execution sends different writers to the same destination without a separate output per experiment.

The original advanced tutorial itself demonstrates the same-patient, multiple-scenario, shared-directory combination. Section 7 contains an executable reproduction; section 9 links baseline and fixed logs.

### 3.2 Output-directory creation races

The baseline checks `isdir` before calling `makedirs`. Two workers can both observe a missing directory; after one creates it, the other can raise `FileExistsError`. A deterministic interleaving test using real filesystem operations reproduced WinError 183. This is not a claim that every uncontrolled concurrent run triggers the race.

### 3.3 Workers remain alive after a batch

In the tested pathos 0.3.5 implementation, leaving the pool context does not clean up workers. Actual worker PIDs remained active after successful batches and ordinary I/O failures. The default pool is cached and reused; this does not establish that every ordinary call creates an additional pool indefinitely.

## 4. Implementation and compatibility

Production changes are confined to [sim_engine.py](../simglucose/simulation/sim_engine.py):

- Allocate filenames in the parent using normalized real destination paths. Unique destinations retain `patient.csv`; duplicates receive input-order `patient__N.csv` names while reserving other tasks' natural filenames.
- Use `os.makedirs(path, exist_ok=True)` to tolerate a directory created by another worker. Permission errors and invalid file destinations still propagate.
- Give each batch a pool with a unique ID, and clear it explicitly in `finally`. This avoids clearing another caller's cached pool. The tradeoff is fresh worker startup for every parallel batch.
- Preserve the `batch_sim(sim_instances, parallel=False)` signature and result order. Add optional `filename` arguments to `sim()` and `SimObj.save_results()`.
- Preserve no-argument calls to custom save methods for unique destinations, including custom non-filesystem persistence with `path=None`. A custom writer participating in duplicate file destinations must accept the optional filename.

The advanced tutorial gains a Windows main guard and separate scenario directories. Regression tests and independent harnesses were added. Patient equations, controller defaults, rewards, Gym reset semantics, and sensor-sharing behavior were not changed.

Repeating a batch updates the same deterministic files. Independently running programs should use separate output directories. Serial execution advances caller-owned objects; process execution advances copies. The contract concerns returned and saved data, not equal parent-object mutation.

## 5. Twenty-four-hour matrix and regression suite

The matrix contains 24 configurations: three patients (`#001` in each age group), two meal scenarios, two controllers, and two seed triplets.

- Scenarios: custom meals `(7,45),(12,70),(18,80)` and RandomScenario.
- Controllers: BB and PID, with `P=0.001,I=0.00001,D=0.001,target=140`.
- Patient-initialization, sensor, and scenario seeds: `(1,2,3)` and `(11,12,13)`; random initial patient glucose enabled.
- Start: 2018-01-01; duration: 24 hours. Three serial/two-worker rounds produce 144 matrix simulations, plus four duplicate-output simulations, totaling 148.
- `assert_frame_equal(..., check_exact=True)` checks indices, columns, dtypes, NaN locations, and values. CSV parsing uses `float_precision='round_trip'`. Every parallel batch must record two actual non-parent worker PIDs.

| Check | Baseline | Final implementation |
| --- | --- | --- |
| Complete pytest suite | 15 passed, 580.50 s | 27 passed, 594 warnings, 610.12 s |
| Numerical/persistence matrix | 267 comparisons; 2 persistence errors; numerical equality | 268 comparisons; 0 errors |
| Added regression suite | Recorded failures for overwritten files, mkdir race, retained workers | 12 passed, 80.16 s |
| Fatal lint E9/F63/F7/F82 | Not used as a baseline conclusion | 0 |

The extra comparison after the fix comes from the recovered second duplicate-output file. Regressions also cover path aliases, reserved filenames, reset/reuse, repeated batches, input order, UI statistics and CSVs, success/error cleanup, and custom persistence. Independent review identified a `path=None` compatibility regression; two tests reproduced it before the fix and passed afterward. See the [review record](evidence/batch-consistency/review.md).

The original full suite used two workers. The final suite used the default pool configuration, Agg, and one OpenBLAS/OMP thread. Their suite durations must not be used as a before/after performance comparison.

## 6. Six patients over ten days: time and output comparison

Patient order: `child#001`, `child#002`, `adolescent#001`, `adolescent#002`, `adult#001`, `adult#002`. All use BB, Dexcom, Insulet, random initial glucose, and daily randomized meals. Patient `i` uses seeds `(100+i,200+i,300+i)` for initialization, sensor, and scenario, with `i=0…5`. Each mode constructs fresh objects.

Each experiment spans 2018-01-01 through 2018-01-11, yielding 4,801 rows including the initial sample, seven columns, and nonzero meal input on all ten dates. Round 1 runs serial then parallel; round 2 reverses that order. Batches do not overlap. Parallel runs use four observed child processes; OpenBLAS/OMP/MKL threads are set to one. There are 24 simulations, 240 patient-days, and 24 CSV files overall.

| Round | Serial (s) | Four processes (s) | Speedup |
| --- | ---: | ---: | ---: |
| 1 | 172.35 | 63.92 | 2.70× |
| 2 | 175.86 | 67.18 | 2.62× |
| Mean | 174.10 | 65.55 | 2.66× |

Timing covers the `batch_sim()` call: simulation, CSV writing, worker startup, returned-data transfer, and cleanup. It excludes module imports, object construction, and comparisons. Dividing total serial time by total parallel time gives approximately 2.66× speedup and 62.35% less wall time. Two runs on one machine do not establish a universal performance guarantee.

**All 60 exact comparisons passed:** 24 memory-to-CSV checks, 12 serial-to-parallel memory checks, 12 serial-to-parallel CSV checks, and 12 cross-round repeatability checks. Maximum absolute error was zero for every column: `BG, CGM, CHO, insulin, LBGI, HBGI, Risk`. All 12 corresponding CSV pairs also had identical SHA256 hashes, establishing byte equality. See the [JSON report](evidence/ten-day-benchmark/report.json) and [run log](evidence/ten-day-benchmark/run.log).

## 7. Minimal output-loss reproduction

Save this complete snippet as `reproduce.py` in a checkout where the target version is installed, then run `python reproduce.py`. It was executed against both the baseline and fixed code. Two different meal experiments should retain two output files.

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

On the baseline both modes print `different trajectories: True` but save only `adolescent#001.csv`. The fixed implementation saves `adolescent#001__1.csv` and `adolescent#001__2.csv`. Separate draft issues cover [output persistence](issues/batch-output-collisions.md) and [pool lifetime](issues/batch-pool-lifecycle.md).

## 8. Reproduce in an isolated environment

Run these commands from the fixed branch's repository root with uv installed. Both output directories must be new; choose another name for each rerun.

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r docs/evidence/batch-consistency/requirements.txt
uv pip install --python .venv/Scripts/python.exe --no-deps -e .
$env:MPLBACKEND = 'Agg'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
.venv/Scripts/python.exe -m pytest tests/test_batch_consistency.py -q
.venv/Scripts/python.exe -m pytest -q -o faulthandler_timeout=240
.venv/Scripts/python.exe examples/verify_batch_consistency.py --output-dir results/new-matrix
.venv/Scripts/python.exe examples/benchmark_ten_days.py --output-dir results/new-ten-day
.venv/Scripts/python.exe -m flake8 simglucose tests examples --count --select=E9,F63,F7,F82 --show-source --statistics
```

To reproduce the original defect, use a separate checkout of the baseline, install that checkout with the same dependencies, and execute section 7. Changing a reported version string is not a substitute for actually loading the baseline code. The matrix harness can also target that checkout through `PYTHONPATH`; original tests must run from that checkout. The [original validation record](serial-parallel-validation.md) contains the baseline commands.

## 9. Evidence, provenance, and limitations

- [Baseline matrix](evidence/batch-consistency/baseline-matrix.json), [fixed matrix](evidence/batch-consistency/fixed-matrix.json), [baseline suite](evidence/batch-consistency/baseline-full-pytest.log), [final suite](evidence/batch-consistency/final-full-pytest.log), [12 regression tests](evidence/batch-consistency/final-regressions.log).
- [Baseline file reproduction](evidence/batch-consistency/issue-baseline-files.log), [fixed file reproduction](evidence/batch-consistency/issue-fixed-files.log), [baseline pool reproduction](evidence/batch-consistency/issue-baseline-pool.log), [fixed pool reproduction](evidence/batch-consistency/issue-fixed-pool.log).
- [Regression source](../tests/test_batch_consistency.py), [matrix harness](../examples/verify_batch_consistency.py), [ten-day harness](../examples/benchmark_ten_days.py), [acceptance audit](evidence/batch-consistency/completion-audit.md).
- Raw CSVs remain locally under `results/ten-day-benchmark-20260930/`, which Git ignores. The harness regenerates them. Archived text logs have only trailing whitespace normalized.

Tested engine SHA256: `76f561f41a34ebd0b0f9c53e61b04d887f8d376beb0059fc27fe5699f0d65c6d`. The initial matrix ran before the fix was committed, so its Git HEAD field still points to the baseline; the source hash identifies the actual tested implementation. The ten-day run used the committed fix. Production code did not change between those validations.

Early simultaneous runs of the default sixteen-worker suite and a harness retaining independent pools exhausted Windows memory/pagefile resources. Those interrupted runs are not counted as passing. Resource-bounded reruns and the final complete suite finished successfully. Logs retain dependency deprecations and Agg notices. Without optional rllab installed, its existing test returns early; 27 passes do not establish that rllab training executed.

Coverage does not include all thirty patients, arbitrary custom controllers, independent unseeded runs, cross-platform/dependency bitwise equality, all worker counts, cross-program locking of one output directory, or crash recovery. The 2.66× figure applies only to section 6. The change does not provide historical output retention across batches or parent-object mutation parity.

## 10. Contribution use

Use this report as supporting documentation, a concise reproduction as the issue body, and the implementation/test summary as the PR body. The persistence finding and pool ownership finding can be discussed separately. The current local implementation commit contains both; if maintainers request separate patches, split them and validate the resulting branches.

The contributor-specific [next steps](contributing-next-steps.zh-CN.md) and [English PR draft](pull-request-batch-consistency.md) accompany this report. Local completion does not imply upstream submission or maintainer acceptance.
