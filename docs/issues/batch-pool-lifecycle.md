# batch_sim retains pathos workers after returning, including when a task fails

## Observation

On baseline commit `a6f6777586c860457a2a1e6cce6d0bd3fd33d1cc`,
`batch_sim(..., parallel=True)` uses `with Pool() as p`, but in the tested
pathos 0.3.5 implementation `ProcessPool.__exit__()` is a no-op. Actual worker
PIDs remain in `multiprocess.active_children()` after both successful and
failed calls. The default pool is cached/reused; this finding does **not**
mean every ordinary call necessarily adds another pool.

Environment: Windows 10 build 19045, Python 3.11.15, pathos 0.3.5,
multiprocess 0.70.19 and dill 0.4.1.

## Reproduction

Use the imports and `task()` function from `batch-output-collisions.md` and
replace the guarded main block with:

```python
if __name__ == '__main__':
    from multiprocess import active_children

    engine.Pool = partial(engine.Pool, nodes=2)
    before = {p.pid for p in active_children()}
    with TemporaryDirectory() as directory:
        root = Path(directory)
        engine.batch_sim([task(root / 'one', 20), task(root / 'two', 50)],
                         parallel=True)
        print('workers left running:',
              {p.pid for p in active_children()} - before)
```

On the baseline this reports two surviving PIDs. The regression suite also
checks the error path by using a regular file as an output directory: the I/O
exception propagates, but workers remain active.

## Suggested ownership rule

Each batch owns a pool with a unique ID and clears it in `finally` on success
or failure. A unique ID is necessary because pathos caches by ID: simply
clearing a shared/default pool can disrupt another caller's pool.

The cost is process startup for each batch instead of implicit cache reuse.
Numerical outputs and the order of returned frames should be unchanged.

Verification: the worker-lifecycle regression tests failed on the original
code and passed with the proposed change. See
`docs/serial-parallel-validation.md` for commands and recorded results.

This is a separate draft because pool ownership can be reviewed independently
of the CSV naming and directory-creation fixes. It has not been posted.
