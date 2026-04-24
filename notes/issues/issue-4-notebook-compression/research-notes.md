# Research Notes: Notebook Compression

## Task Contract

Submission must expose one executable at `/app/run` with exactly:

```bash
./run fit <visible_dir> <artifact_dir>
./run compress <artifact_dir> <input_dir> <compressed_dir>
./run decompress <artifact_dir> <compressed_dir> <recovered_dir>
```

Semantics:
- `fit`: learn/build artifacts from visible corpus only.
- `compress`: one-to-one mapping from each input file to compressed output file.
- `decompress`: reconstruct hidden holdout byte-for-byte with same relative paths.

## Scoring And Validity

Primary metric (raw): `geom_mean_ratio` (lower is better).
Secondary metric: global compression ratio `(artifact_bytes + compressed_bytes)/original_bytes`.

Hard fail conditions include:
- Round-trip is not byte-identical.
- Missing one-to-one attributable compressed outputs.
- Non-regular file types in input/artifact/compressed outputs.
- Missing or non-executable `/app/run`.
- Stage timeout/failure (`fit`, `compress`, `decompress`).

## Resource And Runtime Limits

From task spec and task config:
- CPU only, 16 vCPU
- 32 GiB RAM
- 150 GiB storage
- No internet
- `fit`: 1200s
- `compress`: 1200s
- `decompress`: 600s
- Submission bundle cap: 512 MiB
- Artifact cap: 8 GiB

## Upstream Task Structure (Reference)

Upstream notebook-compression includes:
- `instruction.md`
- `task.toml`
- `job.yaml`
- `oracle.yaml`
- `environment/Dockerfile`
- `environment/workspace/entrypoint.sh`
- `environment/workspace/run`
- `environment/workspace/timer.sh`
- `tests/test.sh`
- `tests/compute_reward.py`
- `tests/scoring_core.py`
- `tests/hidden_test_set_bundle.zip`
- supporting scripts and sources metadata

## Implementation Implications For This Repo

Compared to postgres-sqlite task:
- Notebook compression is not a server protocol task.
- L1 scoring should use compression verifier output parsing, likely `l1_score_mode = compression` equivalent via ratio parser or dedicated output pattern.
- Task should still fit existing episode protocol (`submit_plan`, `submit_subtask`, `advance`) without core API changes.
- Add task-specific image and tests in `tasks/notebook-compression/` with minimal changes to core environment.
