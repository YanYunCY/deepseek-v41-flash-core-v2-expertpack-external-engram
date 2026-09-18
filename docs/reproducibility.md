# Reproducibility contract

## Pin these inputs

For every public result, record:

- model release ID and ModelScope revision;
- release descriptor SHA-256;
- runtime source archive SHA-256 and compiler flags;
- GPU model/architecture, VRAM, host-memory limit, CPU quota, ROCm and driver versions;
- cache sizes, batch/ubatch, context, slot count, thread counts, and I/O mode;
- benchmark script commit and the exact prompt/needle generator version.

The release assets are intentionally not copied into GitHub. Publish the exact descriptor and evidence archive in the ModelScope release, then link both from the GitHub release notes.

## Recommended acceptance sequence

1. Validate the descriptor and all file hashes.
2. Extract the runtime into a fresh build directory and run the offline deployment tests.
3. Start the service on loopback and verify `/health` plus one real chat completion.
4. Run a small smoke benchmark before any long-context run.
5. Run the concurrency benchmark twice per setting; report normal `stop`, `length` cutoffs, and invalid requests separately.
6. For long-context runs, verify raw `/tokenize` length, response `usage`, all needle values, SSE `[DONE]`, and `finish_reason`.
7. Archive raw JSON and logs privately, then publish only redacted summaries and hashes.

## Current evidence

The originating ModelScope materials report completed 16K and 64K retrieval checks on one AMD `gfx942` / ROCm 7.2.3 instance. A 1M context slot was allocatable, but a complete 1M-token quality result was not accepted as passed. Treat all speed numbers as machine- and workload-specific observations, not SLAs.

Reference release pages:

- [ModelScope model repository](https://modelscope.cn/models/Yanyunawa/DeepSeek-V4.1-Flash-MXFP4-GGUF)
- [Deployment article](https://modelscope.cn/learn/436737)
- [Performance article](https://modelscope.cn/learn/436741)
- [Acceptance notebook](https://modelscope.cn/gallery/Yanyunawa/amd-deepseek-v41-deployment)
