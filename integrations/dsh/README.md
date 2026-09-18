# DSH integration

The adapter is an optional compatibility layer. It listens on `127.0.0.1:48242`, loads the official V4.1 encoder from a path supplied at runtime, and translates DSH/OpenAI-compatible chat requests to the native local server at `127.0.0.1:48241`.

```bash
python3 integrations/dsh/dsh_api_adapter.py \
  --encoder /srv/dsv41/capability/encoding.py \
  --upstream http://127.0.0.1:48241 \
  --port 48242
```

The Windows PowerShell helpers maintain a loopback-only local forward. They require an existing SSH config entry, strict host-key checking, and an explicit relay host. They do not copy or generate private keys.

For the Linux supervisor, set the relay explicitly before starting it:

```bash
export DSV41_RELAY_HOST=relay.example.invalid
python3 integrations/dsh/start-dsh.py
```

For the Windows installer, pass `-ExpectedRelayHost` explicitly. The public scripts intentionally fail when no relay host is supplied.

The relay host, user, ports, and key paths are deployment-specific. Set them explicitly; the public templates intentionally do not contain the original infrastructure address or public key.

The adapter path is experimental until it has been tested end to end with the official encoder, real llama-server, thinking modes, tool calls, cancellation, and tunnel reconnects.
