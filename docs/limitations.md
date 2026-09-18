# Limitations and honest claims

- The included offline tests use synthetic HTTP streams and temporary fixtures. They do not validate a real AMD GPU, ROCm installation, ModelScope download, or DSH client session.
- The deployment path was validated on AMD `gfx942` with ROCm 7.2.3. Other GPUs, drivers, memory sizes, and filesystem layouts need their own acceptance run.
- 64K actual input retrieval was completed in the originating evidence. A 1M context allocation and an in-progress million-token request must not be described as a passed 1M-token evaluation.
- The DSH adapter still needs an end-to-end run with the official encoder, native server, SSH relay, thinking modes, tool calls, cancellation, and slot release.
- The scripts do not reproduce all official cloud-service features, multimodal input, or cloud search.
- Performance numbers depend on prompt set, cache warmth, batch shape, output cap, and process history. They are observations, not guarantees.
- The model weights and any modified third-party runtime are outside this repository and retain their own licenses.
