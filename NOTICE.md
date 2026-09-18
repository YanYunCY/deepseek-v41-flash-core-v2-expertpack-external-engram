# Third-party and release boundary

This repository contains original deployment, benchmark, and integration scripts. It is not a model-weight repository.

- DeepSeek-V4.1-Flash weights and their tokenizer/encoder are governed by the upstream DeepSeek license.
- Any llama.cpp-derived runtime or Expert Streaming/Engram implementation must retain its own upstream copyright and license notices.
- DSH is a separate product and package; this repository only contains an optional local adapter and helper scripts.
- ModelScope is used as the artifact/release host for large model files and exact SHA-256 manifests.

Before publishing a release, verify the license files shipped with every runtime source archive and link the exact model revision used by the release descriptor.
