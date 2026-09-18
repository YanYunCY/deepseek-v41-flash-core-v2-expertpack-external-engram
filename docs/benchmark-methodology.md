# Benchmark methodology

## Validity classes

Every request belongs to one of three classes:

- `normal_stop`: stream completed, content was non-empty, usage was internally consistent, and the server reported `stop`.
- `length_cutoff`: the request is usable for throughput accounting but reached the configured output limit; it is not a quality pass.
- `invalid`: HTTP, SSE, usage, content, tokenizer, or needle validation failed.

Never combine these classes into a single quality claim. Throughput may report valid `stop` and `length` samples separately.

## Long-context checks

The long benchmark reserves output/template space from the configured context capacity. It obtains a raw token count from `/tokenize`, records the expected needle offsets, and then requires all of the following before `quality_pass=true`:

1. server-reported prompt usage covers the raw token count;
2. all deterministic needle values are present in the answer;
3. SSE contains a complete terminal event;
4. completion usage is present and totals add up;
5. `finish_reason` is `stop`;
6. the answer is non-empty.

This is a retrieval and service-integrity check, not a general long-context capability evaluation.

## Concurrency checks

Use at least two rounds per concurrency setting. Report wall-clock batch time, effective output tokens per second, TTFT percentiles, request counts, normal stops, length cutoffs, and invalid requests. A high aggregate token rate with many truncated or invalid requests is not a successful serving configuration.
