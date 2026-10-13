# Data inputs

The raw Endomondo/FitRec dataset and workout-level preprocessed sequences are
not included in this repository.

The analysis scripts expect two directories, each containing
`sequence_index.csv` and the referenced `sequences-*.npz` shards:

- development/validation population sequences;
- locked user-disjoint test sequences.

Each shard must contain the arrays `elapsed_seconds`, `speed_kmh`,
`heart_rate_bpm`, and `altitude_m`. The index records the shard, offset,
sequence length, pseudonymous user/workout keys, chronological position, role,
timestamp, split, and signal-coverage fields.

The official FitRec dataset description and access information are available
at <https://cseweb.ucsd.edu/~jmcauley/datasets/fitrec.html>. Users are
responsible for complying with the source terms and applicable institutional
data-governance requirements.

Input index hashes used for the included results are recorded in the experiment
manifests under `results/`.
