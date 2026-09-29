# Issue draft index

The investigation produced two independently reviewable drafts:

1. [Batch output collisions and directory creation race](batch-output-collisions.md):
   seeded numerical trajectories agree, but same-patient scenarios lose saved
   results; a directory-creation interleaving can also raise an exception.
2. [Pool lifetime](batch-pool-lifecycle.md): pathos workers remain alive after a
   batch, including its ordinary error path.

Use the first draft for the original serial/process output-consistency concern.
The second separates a related resource-ownership fix from the persistence issue.
Both drafts contain executable reproductions and remain unpublished.

See [validation report](../serial-parallel-validation.md) for the tested
versions, comparison matrix, regression evidence and compatibility boundaries.
