# DeepSeek-V4.1-Flash: Core v2 + ExpertPack + External Engram

可复现的 DeepSeek-V4.1-Flash 社区部署、基准测试和本地 DSH 接入工具集，重点针对 AMD `gfx942` / ROCm 主机。

This repository contains the reusable engineering around a community deployment of DeepSeek-V4.1-Flash:

- a manifest-driven deployment launcher with file, archive, and process-safety checks;
- long-context and concurrency benchmarks that validate SSE completion, usage accounting, needle retrieval, and finish reasons;
- an optional loopback-only OpenAI-compatible adapter for DSH;
- a conservative benchmark stage controller that never overwrites the runner or sends a second stop signal.

## What this is — and is not

This is the code and reproducibility layer. It does **not** contain model weights, private credentials, raw user conversations, or a complete llama.cpp source tree. The model assets are roughly 500 GB and remain in the linked ModelScope release.

The project is a community deployment, not an official DeepSeek distribution. A successful 1M-token context allocation is not the same as a completed 1M-token quality test. Current evidence has completed 16K and 64K retrieval checks on one AMD `gfx942` / ROCm 7.2.3 instance; hardware, driver, runtime, and prompt changes require new measurements.

## Quick start

The deployment launcher expects a local data root containing the release assets and a release descriptor. The descriptor is intentionally kept separate from this code repository because it contains release-specific paths and hashes.

```bash
python3 deployment/deploy.py prepare \
  --root /srv/dsv41 \
  --data-root /srv/dsv41-data \
  --release /srv/releases/20260916/release.json

python3 deployment/deploy.py start --root /srv/dsv41
python3 deployment/deploy.py status --root /srv/dsv41
python3 deployment/deploy.py stop --root /srv/dsv41
```

### Connect a running instance to Windows DSH

The model launcher and the DSH connection are separate lifecycles. Starting
`deploy.py` only makes the local model API available on `127.0.0.1:48241`; it
does not create the adapter or an SSH tunnel. A new ModelScope instance also
does not inherit `/root/dsv41/integrations/dsh` from an earlier instance.

For the public, redacted templates, copy `integrations/dsh/` to the inference
host and provide the deployment-specific relay explicitly:

```bash
export DSV41_RELAY_HOST="relay.example.invalid"
python3 integrations/dsh/start-dsh.py
```

The model-specific bundle, including the official V4.1 encoder, is published
under `deployment/dsh/` in the linked ModelScope model. On a fresh ModelScope
instance, download the four runtime files after the model service is ready:

```bash
python3 - <<'PY'
from pathlib import Path
from modelscope.hub.file_download import model_file_download

repo = "Yanyunawa/DeepSeek-V4.1-Flash-MXFP4-GGUF"
out = Path("/root/dsv41/dsh-integration")
out.mkdir(parents=True, exist_ok=True)
for name in ("start-dsh.py", "dsh_api_adapter.py", "remote-connect.py", "encoding.py"):
    source = Path(model_file_download(
        model_id=repo, file_path=f"deployment/dsh/{name}"))
    (out / name).write_bytes(source.read_bytes())
PY
python3 /root/dsv41/dsh-integration/start-dsh.py
```

The instance must already contain the user-managed
`/mnt/workspace/.dsv41-connection/dmit_ed25519` and `known_hosts` files; they
are never stored in GitHub or ModelScope. On Windows, the DSH task forwards
`127.0.0.1:48241` through the relay. Verify the complete path before opening a
new DSH session:

```powershell
Invoke-RestMethod http://127.0.0.1:48241/v1/models
```

The verified path supports text, thinking configuration, and function-tool
round trips. Image input, cancellation/release under load, and every official
cloud feature remain outside this adapter's acceptance scope.

Use `examples/release.json.example` only as a schema reference. A live release must use a versioned descriptor whose every asset has the exact ModelScope size and SHA-256 value.

Run the offline tests with the Python standard library only:

```bash
python3 -m unittest discover -s deployment -p 'test_*.py'
python3 -m unittest discover -s benchmarks -p 'test_*.py'
python3 -m unittest discover -s integrations/dsh -p 'test_*.py'
python3 -m unittest discover -s integrations/benchmark-control -p 'test_*.py'
python3 -m unittest discover -s tools -p 'test_*.py'
```

Before a public commit, run `python3 tools/public_audit.py .`. It catches common weights, transfer-payload, private-key, and credential-file mistakes; it is deliberately only a guardrail, so review generated JSON and documents manually too.

The tests use synthetic HTTP streams and filesystem fixtures. They do not claim to replace an AMD/ROCm acceptance run.

## Evidence snapshot

| Check | Status | Meaning |
| --- | --- | --- |
| Deployment/hash/receipt logic | 9 offline tests pass | The safety and planning paths are locally regression-tested. |
| Benchmark/SSE/usage logic | 14 offline tests pass | Invalid streams and accounting errors are rejected. |
| DSH adapter logic | 8 offline tests pass | Synthetic tool/thinking/SSE cases are covered. |
| Benchmark stage control | 12 offline tests pass | Stop/report simulations protect the runner state. |
| 16K and 64K retrieval on `gfx942` | Passed in originating evidence | A bounded retrieval check, not a general capability score. |
| 1M context | Capacity allocated; quality run not accepted as passed | Do not advertise this as a completed 1M-token evaluation. |

## Repository layout

| Directory | Purpose |
| --- | --- |
| `deployment/` | Release validation, safe source extraction, receipt reuse, and launcher process ownership. |
| `benchmarks/` | Long-context and concurrency measurement with explicit validity classes. |
| `integrations/dsh/` | Loopback adapter, tunnel supervisor, and Windows helper scripts. |
| `integrations/benchmark-control/` | Safe stop-after-current-result and stage-report tooling. |
| `docs/` | Architecture, measurement rules, security guidance, and limitations. |
| `examples/` | Redacted release descriptor templates. |

See [`docs/allocation-profiles.md`](docs/allocation-profiles.md) for the measured interactive, throughput, and long-context profiles.

## Artifact split

Keep the public Git repository small and reviewable:

1. GitHub: source, tests, docs, examples, and release metadata schemas.
2. ModelScope: model weights, ExpertPack/Engram sidecars, exact release manifests, and large evidence archives.
3. Private storage: raw prompts, full server logs, credentials, SSH host details, and one-off terminal installers.

The original deployment materials and benchmark articles are available from the ModelScope model/release pages referenced in `docs/reproducibility.md`. The full model-page README is mirrored at [`docs/modelscope-model-README.md`](docs/modelscope-model-README.md) so the deployment description and the public source remain reviewable together.

## Security posture

The launcher and adapter default to loopback addresses. The tunnel scripts require an explicit relay host and existing key/known-host files. Never commit private keys, `known_hosts`, raw logs, or copied terminal payloads. Review generated benchmark JSON before publishing because it may contain prompts, absolute paths, or infrastructure metadata.

## License boundary

Original scripts in this repository are released under the MIT license in `LICENSE`. That license does not re-license DeepSeek weights, llama.cpp-derived runtime code, DSH packages, or other third-party components. See `NOTICE.md` and the upstream licenses before redistributing any model or runtime artifact.

## Status

The deployment path and the DSH text/tool round trip have been re-run end to
end with the official encoder, native server, tunnel, and Windows client. The
adapter remains intentionally limited: image input, cancellation under load,
and cloud-only features are not claimed. See `docs/limitations.md` and
`integrations/dsh/README.md`.

