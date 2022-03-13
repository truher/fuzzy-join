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

## Dependencies

I modified some of the libraries, look in requirements.txt.

* git+https://github.com/truher/python-string-similarity.git avoids matching
whitespace-only or empty strings
* git+https://github.com/truher/string_grouper.git adds some config and
makes string-series detection more robust
* git+https://github.com/truher/red_string_grouper.git allows configuration
of the TfidfVectorizer and joins multiple matchers with an outer join, so that
match methods can disagree more (previously it was inner, i.e. all matchers
needed to more-or-less agree).

