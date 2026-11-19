# Validation status

Validated before initial publication:

- 13 standard-library offline tests passed, including real mock HTTP streaming,
  token accounting, thermal aborts, CPU mismatch, interrupted/failed result
  handling, regression comparison, and mocked full shell orchestration for
  successful quick runs and failed model startup.
- `bash -n run.sh` passed.
- Official Docker Compose v2.39.4 validated both profiles with
  `--profile tests --profile inference config --quiet`.
- Default Hugging Face model revision was checked against its public model API.

**Not yet validated on GPUs:** this development workspace has no Docker Engine
or GPU test runtime. Container build, CUDA memory patterns/GEMM, NCCL, vLLM model
startup and real hardware performance have not been executed here. Offline CI
or configuration validation must not be interpreted as target-machine health.

On the target Debian machine, run `./run.sh preflight`, then reserve a GPU
maintenance window and run `./run.sh quick`. Preserve the result directory;
only that evidence can establish whether the selected workload works there.
