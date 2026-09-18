# Deployment launcher

`deploy.py` is the reference launcher for a release described by a JSON manifest. It is designed for large, immutable model data roots and a separate runtime root.

## Commands

```bash
python3 deployment/deploy.py prepare --root /srv/dsv41 --data-root /srv/dsv41-data --release /srv/release.json
python3 deployment/deploy.py start --root /srv/dsv41
python3 deployment/deploy.py status --root /srv/dsv41
python3 deployment/deploy.py stop --root /srv/dsv41
```

`prepare` verifies assets, extracts only the expected runtime tree, builds the pinned server, and records a receipt. `start` reuses the receipt and does not silently download or rebuild. `stop` only signals a process whose PID and recorded Linux start time still match.

## Release descriptor

Start from [`../examples/release.json.example`](../examples/release.json.example), fill in the exact sizes and lower-case SHA-256 values, and keep the descriptor beside the artifact release. Do not commit a descriptor containing private URLs or credentials.

The launcher expects one core asset, two manifests, the exact ExpertPack/Engram sidecars, a runtime source archive, and a per-file source hash JSON. The exact role counts are validated before any build step.

## Environment

The tested path uses Linux, Python 3.10+, a C/C++ toolchain, CMake, ModelScope SDK, and HIP/ROCm for `gfx942`. The service is intentionally loopback-only by default. A first prepare requires hundreds of gigabytes of free space for the external model assets; the code repository itself is small.

The launcher is not a generic llama.cpp installer. It assumes the release runtime exposes the external ExpertPack/Engram command-line options used by the descriptor and preserves the runtime archive's own license notices.
