# Completion audit

Contract: same explicit configurations/seeds, deterministic controller, same
runtime; equal returned data and complete, deterministic per-batch persistence.

| Requirement | Authoritative evidence |
| --- | --- |
| Original version identified | Baseline commit a6f6777586c860457a2a1e6cce6d0bd3fd33d1cc; baseline-matrix.json source SHA256 |
| Reproducible Windows/Python stack | requirements.txt; platform/packages in both matrix JSON files |
| Baseline actually exercised | baseline-full-pytest.log: 15 passed; baseline-matrix.json: actual child PIDs, only duplicate-file errors |
| Numerical equality | Matrix harness uses exact comparisons for index, seven columns, dtype and NaNs, not tolerances |
| Multiple patients/scenarios/controllers/seeds | 24 explicit configs recorded in matrix JSON; three age cohorts, custom/random meals, BB/PID and two seed triplets |
| Repetition and resetting | Matrix repeats three times; test_reset_and_reuse_keep_duplicate_outputs_stable |
| Actual multiprocessing | Non-parent, two distinct worker PIDs checked per matrix batch; regression tests also check PID identity |
| In-memory and CSV equality | Matrix CSV round-trip checks; test_repeated_batches_match_in_memory_and_on_disk |
| Duplicate destinations and aliases | Baseline failing logs; fixed duplicate/alias/reserved-name tests; final duplicate report includes both files |
| Directory-creation race | Deterministic real-filesystem interleaving RED→GREEN; concurrent integration runs in new directories |
| Pool cleanup on success/failure | baseline-pool-lifecycle.log RED; regression active_children assertions GREEN |
| UI-level reports | test_high_level_ui_produces_identical_reports_and_patient_csvs |
| Custom persistence compatibility | Independent reviewer reproduced regression; review-custom-persistence-red.log; both modes GREEN in final-regressions.log |
| Standard filenames and input order | Distinct-directory legacy-name test; zipped config/result exact comparisons and deterministic duplicate filename test |
| Windows example | Main guard in advanced_tutorial.py; advanced-tutorial.log records successful execution |
| No model/controller/reward change | Production diff limited to simulation/sim_engine.py |
| Final complete test suite | final-full-pytest.log: 27 passed, 594 warnings, 610.12 seconds; default pool configuration |
| Fatal lint checks | flake8.log: 0 for E9,F63,F7,F82; git diff --check |
| Independent review | review.md records findings and the single fix pass |
| Issue-ready minimal examples | issue-baseline-files.log, issue-fixed-files.log, issue-baseline-pool.log, issue-fixed-pool.log from verbatim fenced snippets |
| Requested deliverables exist | Plan, tests/test_batch_consistency.py, examples/verify_batch_consistency.py, docs/serial-parallel-validation.md, docs/issues/serial-parallel-consistency.md index and both drafts |
| Honest limitations | Report records interrupted resource-exhausted runs, numerical-library warnings, optional rllab absence and untested platforms |
| No external publication | Only local files and local Git operations; neither issue posted |

Final artifacts inspected on 2026-09-30: 24 configurations, 148 simulations,
268 exact comparisons, zero errors; source SHA256 matches the final engine.
Serial PID identity and two distinct non-parent worker PIDs in each parallel
batch were checked. The final regression suite has 12 passes and the final
complete suite has 27 passes. All deliverables are local; issues remain unpublished.
Archived logs differ from raw results only by trailing-whitespace normalization.
