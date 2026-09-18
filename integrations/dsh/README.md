# DSH integration

This directory contains the public, redacted compatibility layer. It is
separate from model deployment: the native server must be healthy on
`127.0.0.1:48241` before the adapter or tunnel can be useful. A new remote
instance does not automatically contain these files.

The adapter listens only on `127.0.0.1:48242`, loads the official V4.1 encoder
from a path supplied at runtime, and translates DSH/OpenAI-compatible chat
requests to the native local server at `127.0.0.1:48241`.

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

For a fresh ModelScope instance using the published model-specific bundle,
download the adapter, supervisor, encoder, and starter together. Downloading
only `deployment/20260916/deploy.py` is insufficient and will produce
`can't open file .../dsh-integration/start-dsh.py`:

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

The model-specific starter expects the existing user-managed key and host-key
files under `/mnt/workspace/.dsv41-connection/`. It does not generate or copy
credentials. The starter is idempotent: a file lock prevents duplicate
adapters and the tunnel supervisor reconnects with bounded backoff. State and
logs should live on the instance's local disk when the persistent workspace
quota is tight.

After the starter reports `DSH_ADAPTER_READY`, verify from Windows:

```powershell
Invoke-RestMethod http://127.0.0.1:48241/v1/models
```

Then start a new DSH session. Existing DSH sessions cache their provider
settings and should be closed before testing a newly connected route.

The checked-in DSH configuration writes `thinking: enabled` explicitly (do not
rely on `reasoningEffort` alone). It keeps the model's 1,048,576-token context
and 262,144-token output capability,
while limiting the default thinking phase to 2,048 tokens for `low` and 16,384
tokens for `high`/`max`. Thinking and the final answer share the native output
budget; the cap prevents a difficult prompt from spending the whole turn in
reasoning before emitting an answer. The headless client prints reasoning to
stderr and the adapter forwards it as `reasoning_content` SSE deltas for UIs
that render reasoning blocks. If an OpenAI-compatible caller omits `thinking`,
the adapter follows the V4.1 Flash API default (`enabled`, `high`) and
derives the same bounded budget. Thinking and the final answer share the
native output budget; the cap prevents a difficult prompt from spending the
whole turn in reasoning before an answer is emitted.

The first visible event is the assistant role marker. The model can still be
silent while the native server performs prompt prefill; this is proportional
to uncached input length. Official DeepSeek requests use `stream: true` and
deliver reasoning in `choices[0].delta.reasoning_content`, but the cloud
service has stronger prompt caching and scheduling than a single local slot.

For the Windows installer, pass `-ExpectedRelayHost` explicitly. The public scripts intentionally fail when no relay host is supplied.

The relay host, user, ports, and key paths are deployment-specific. Set them explicitly; the public templates intentionally do not contain the original infrastructure address or public key.

The text path and a function-tool round trip have been tested end to end with
the official encoder, real llama-server, tunnel, and Windows DSH client. The
short verification requests did not emit visible reasoning text, so this is
not a claim that every thinking-stream presentation is verified. Image input,
cancellation under load, and cloud-only features remain unsupported by this
local adapter.
