# Allocation profiles

The current evidence does not support one universal allocation. The useful unit is a profile selected by workload.

Hardware reference: AMD `gfx942`, approximately 191.69 GiB visible VRAM, and a 200 GiB host-memory limit. Headroom below is calculated from the observed process peak, not a hardware guarantee.

## Recommended profiles

| Workload | GPU cache | Host cache | Batch | Context / slots | Observed result | Recommendation |
| --- | ---: | ---: | ---: | --- | --- | --- |
| Interactive short chat | 172 GiB | 96 GiB | 128 | 4096 / 1 | 17.07 warm decode tok/s; 180.90 GiB VRAM peak | Default balanced profile; about 10.8 GiB VRAM headroom. |
| Host-rich short chat | 172 GiB | 128 GiB | 128 | 4096 / 1 | 17.56 warm decode tok/s; 138.39 GiB RSS peak | Optional; only when at least ~60 GiB host headroom is genuinely available. |
| Interactive concurrency | 160 GiB | 96 GiB | 512 | 8192 total / 2 | 8.15 aggregate tok/s; TTFT p50 16.2 s | Better latency/throughput compromise than 4 or 8 slots. |
| Batch throughput | 160 GiB | 96 GiB | 512 | 32768 total / 8 | 11.15 aggregate tok/s; all 16 requests stopped normally | Highest tested throughput; TTFT p50 36.4 s, so not interactive. |
| 64K-class long input | 152 GiB | 96 GiB | 512 | 1,048,576 / 1 | 54.7 prefill tok/s; 64K retrieval passed | Safe long-context starting point; about 17.6 GiB VRAM headroom. |
| 16K prefill experiment | 136 GiB | 96 GiB | 2048 | 1,048,576 / 1 | 99.8 prefill tok/s; independent 16K needle passed | Candidate for controlled experiments, not a general default. |

## What not to use as a default

`180 GiB GPU / 96 GiB host` reached 17.87 warm decode tok/s, only about 4.7% above the balanced 172/96 profile, while observed VRAM headroom fell to roughly 2.4 GiB. That margin is too small for allocator variance, a different prompt, or another process.

For 16K long input, batch 1024 reached 105.8 prefill tok/s but used 189.37 GiB VRAM, leaving about 2.3 GiB. It is a useful upper-bound experiment, not a resilient service setting.

## Commands

Balanced interactive service:

```bash
python3 deployment/deploy.py start --root /srv/dsv41 \
  --gpu-cache-gib 172 --host-cache-gib 96 \
  --batch 128 --context 4096 --parallel 1
```

Throughput batch service:

```bash
python3 deployment/deploy.py start --root /srv/dsv41 \
  --gpu-cache-gib 160 --host-cache-gib 96 \
  --batch 512 --context 32768 --parallel 8
```

64K-class long input:

```bash
python3 deployment/deploy.py start --root /srv/dsv41 \
  --gpu-cache-gib 152 --host-cache-gib 96 \
  --batch 512 --context 1048576 --parallel 1
```

## How to tune further

1. Fix the workload shape first: short chat, 16K/64K long input, or concurrent slots.
2. Sweep batch sizes before increasing cache; larger batch improved long-input prefill much more than increasing GPU cache.
3. Keep at least 8–12 GiB observed VRAM headroom for a service profile; reserve more for long runs and cold starts.
4. Repeat each candidate at least three times, separating cold-start, first-request, and warm-request measurements.
5. Score only valid requests. Report normal `stop`, `length` cutoffs, invalid requests, TTFT percentiles, and total throughput separately.
6. Re-run the matrix after changing ROCm, runtime source, driver, context, or prompt set.

The figures above are observations from the current evidence bundle. They are not a promise that another GPU, filesystem, driver, or model revision will have the same optimum.
