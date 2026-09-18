# Benchmarks

`extended-benchmark.py` measures a local OpenAI-compatible endpoint without treating every HTTP 200 as success.

It supports:

- deterministic long-context documents with three needle values;
- raw `/tokenize` calibration and context-budget accounting;
- streamed chat completion capture with SSE and usage validation;
- repeated concurrency runs with normal-stop, length-cutoff, and invalid classes.

Example shape:

```bash
python3 benchmarks/extended-benchmark.py concurrency \
  --base-url http://127.0.0.1:48241 \
  --concurrency 4 --rounds 2 --outdir ./runs/concurrency-4
```

Use the real service only after the deployment smoke check. Do not publish raw request/response folders without checking for prompts, file paths, or infrastructure metadata.
