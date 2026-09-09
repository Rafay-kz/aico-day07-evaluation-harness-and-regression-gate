# Day 7 controlled regression proof

Temporarily weaken retrieval with a reversible test-only setting, run the same gate, then restore.

Change: `top_k 5 → 1 → 5 (mode=bm25)`

| Stage | top_k | Verdict | Exit code | Hit@K | MRR |
|---|---:|---|---:|---:|---:|
| normal approved | 5 | PASS | 0 | 1.0000 | 0.9286 |
| weakened retrieval | 1 | FAIL | 1 | 0.8571 | 0.8571 |
| restored approved | 5 | PASS | 0 | 1.0000 | 0.9286 |

## Failed thresholds on weakened run

- hit_at_k 0.8571 < threshold 0.9500
- mrr 0.8571 < threshold 0.9000
- refusal_accuracy 0.8333 < threshold 0.8500

Proof passed: **True**

Required: normal PASS, weakened FAIL with non-zero exit, restored PASS.

