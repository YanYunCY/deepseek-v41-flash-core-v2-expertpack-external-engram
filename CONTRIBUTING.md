# Contributing

Keep this repository focused on reusable code and evidence discipline.

Before opening a pull request:

1. Run every unittest suite listed in `README.md`.
2. Add or update a synthetic test for security-sensitive behavior and failure paths.
3. Do not add model weights, private credentials, raw prompts, unredacted logs, or host-specific keys.
4. State the tested Python/OS/runtime versions and whether a check is offline or real-hardware.
5. Preserve the distinction between normal stops, length cutoffs, invalid requests, capacity allocation, and completed quality checks.

Changes to release descriptors, runtime source archives, or model links should include their exact revision and SHA-256 values in the release notes rather than relying on a mutable `latest` pointer.
