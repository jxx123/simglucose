# Preserve batch simulation outputs and clean up owned process pools

## Summary

This patch addresses batch output persistence and process-pool ownership. Seeded numerical trajectories already matched between serial and process execution in the tested baseline; this is not a claim of reproduced solver nondeterminism.

- Allocate deterministic, collision-free per-batch CSV names in the parent, preserving legacy filenames for unique destinations.
- Tolerate concurrent output-directory creation using `exist_ok=True`.
- Give each batch a uniquely identified pathos pool and clear it on success or ordinary task failure.
- Preserve custom no-argument save methods where no filename override is needed, including non-filesystem persistence with `path=None`.
- Add a Windows main guard and separate scenario directories to the advanced tutorial.
- Add regression tests, exact-comparison harnesses, and bilingual reports.

## Reproduced baseline behavior

At `a6f6777586c860457a2a1e6cce6d0bd3fd33d1cc`, two different scenarios for one patient in the same output directory return two different frames but leave only one CSV in both modes. A deterministic filesystem interleaving reproduces the mkdir race. With pathos 0.3.5, worker PIDs remain alive after successful and failing batches; the default pool is cached, so this is not a claim of an additional pool on every ordinary call.

Executable reproductions are in the output-persistence and pool-lifetime drafts under `docs/issues/`. This branch contains both fixes; their findings are documented separately for review.

## Validation

Tested on Windows 10 build 19045, Python 3.11.15, with pinned dependencies:

- Baseline repository suite: 15 passed.
- Final repository suite: **27 passed**, including **12 added regression cases**; 594 warnings retained in the log.
- Twenty-four-hour matrix: **24 configurations, 148 simulations including duplicate-output cases, 268 exact comparisons, zero errors**, with two observed worker PIDs per parallel batch.
- Ten-day benchmark: **six patients, two rounds of serial/four-process execution, 60 exact comparisons, zero errors**. All corresponding CSV pairs were byte-identical.
- Ten-day mean times: **174.10 s serial**, **65.55 s with four processes**, or **2.66×** on this machine. This compares the fixed implementation's modes, not patch-versus-baseline speed.
- Fatal flake8 checks (`E9,F63,F7,F82`) passed.

Commands, exact versions, limitations, and evidence links are in `docs/serial-process-report.en.md`; a Chinese counterpart is at `docs/serial-process-report.zh-CN.md`.

## Compatibility and tradeoffs

The model, controller defaults, rewards, reset behavior, and returned result order are unchanged. Unique output destinations retain `patient.csv`; duplicates use deterministic `patient__N.csv` names. Rerunning a batch updates the same files; separate concurrent programs should use separate directories.

Explicit pool ownership incurs startup for every parallel batch. A custom writer involved in duplicate filesystem destinations must accept the optional filename. Serial and process modes still differ in whether caller-owned objects advance; the equality contract applies to returned and saved data.

Coverage is limited to the recorded environment/configurations. Optional rllab training was not exercised, and no cross-platform bitwise guarantee or general speedup claim is made.
