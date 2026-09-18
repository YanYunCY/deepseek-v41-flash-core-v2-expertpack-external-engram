# Public release checklist

- [ ] Replace the example release descriptor with a versioned, exact descriptor in the artifact host.
- [ ] Confirm every public file has an appropriate license or attribution notice.
- [ ] Run all offline unittest suites on the release commit.
- [ ] Run `python tools/public_audit.py .` and resolve every finding.
- [ ] Run a secret scan and inspect Base64, JSON, notebooks, logs, and generated documents manually.
- [ ] Confirm no private key, token, raw prompt, host-specific IP, or unredacted SSH host key is present.
- [ ] Link the ModelScope model/release and evidence archive from the GitHub release.
- [ ] State the tested GPU/ROCm matrix and clearly separate passed checks from allocated capacity or in-progress runs.
- [ ] Tag a release only after the README quick-start has been tested from a clean checkout.
