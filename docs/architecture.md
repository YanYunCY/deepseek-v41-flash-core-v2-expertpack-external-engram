# Architecture

## Runtime path

```text
release.json + ModelScope assets
          |
          v
deployment/deploy.py
  - validate descriptor and paths
  - download/verify assets
  - verify source archive and per-file hashes
  - build the pinned runtime
  - launch only the owned llama-server process
          |
          v
127.0.0.1:48241 native model API
          |
          +--> benchmarks/extended-benchmark.py
          |
          +--> integrations/dsh/dsh_api_adapter.py :48242
                         |
                         +--> optional SSH tunnel / local DSH client
```

The deployment launcher treats the model data root as immutable input. Runtime state, receipts, the build directory, and process metadata live under the separate `--root` directory. `stop` checks both the recorded PID and Linux process start time before sending a signal.

The adapter is deliberately not a model server. It loads the official encoder supplied by the deployment, translates DSH/OpenAI-compatible messages into native llama-server requests, enforces the context budget, and streams parsed content back over SSE.

## Integrity model

The release descriptor identifies every model/runtime asset by relative path, size, role, and SHA-256. The launcher rejects traversal, whitespace paths, unexpected source archive members, and incomplete ExpertPack/Engram layouts. A receipt can be reused only when the file stat record still matches; a changed file is hashed again.

## Experimental boundary

The benchmark-control scripts are independent of model startup. They only inspect a pre-existing benchmark layout, accept a contiguous verified result prefix, and send at most one signal to the runner. They never stop the model server.
