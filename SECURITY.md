# Security policy

## Safe defaults

- Keep the model server and DSH adapter bound to `127.0.0.1` unless an independently reviewed reverse proxy is in front of them.
- Use an explicit SSH `known_hosts` file and a dedicated forwarding-only relay account.
- Keep private keys, raw logs, benchmark prompts, and runtime state outside the repository.
- Treat release descriptors and source-hash manifests as integrity controls; do not replace them with unverified downloads.

## Reporting

Please do not open a public issue containing credentials, private prompts, host addresses, or unredacted logs. Contact the repository owner privately with a minimal reproduction and the affected commit or release ID.
