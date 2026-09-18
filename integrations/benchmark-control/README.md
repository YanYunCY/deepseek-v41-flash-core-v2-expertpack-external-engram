# Benchmark stage control

These scripts are safe operational helpers for an already-running Linux benchmark:

- `stage_report.py` reads a contiguous, fully written result prefix and emits cautious statistics;
- `stop_after.py` waits for a complete prefix and validates the runner identity before one signal;
- `stop-after-current.py` is the pidfd-aware controller used by the simulation tests.

They do not start inference, rewrite the dataset, mutate the runner, stop the model server, or send a second signal after a prior attempt. The actual Linux controller requires `/proc`, `fcntl`, and the benchmark's own result layout. The unit tests mock those pieces and are Windows-runnable simulations.
