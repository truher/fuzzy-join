# fuzzy-join

A small pipeline to join two tables together based on a fuzzy match
of a few keys.  One of the full tables is not small enough to fit into RAM,
so where possible, it is read as a chunked scan, broadcast-joined to the
other table.

## Implementation details

The process has several stages, orchestrated with the "doit" library,
which is pretty simple.  By far the slowest stage is the initial candidate
generation, which computes cosine similarity between pairs.

The python multiprocessing apply\_async method is used,
because it doesn't require knowledge of the length of the input (as map does).

The chunk reader also peeks at the pool task queue length, to avoid getting
ahead of the workers, overfilling the parent process.
