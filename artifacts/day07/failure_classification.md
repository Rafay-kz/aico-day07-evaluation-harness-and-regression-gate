# Day 7 failure classification

Every failed case has exactly one primary type: chunking, retrieval, prompt, citation, refusal, or evaluator.

| Case ID | Split | Category | Observed failure | Primary type | Reason |
|---|---|---|---|---|---|
| G13 | train | multi_chunk | answer | prompt | missing critical facts: sixty days |
| G14 | development | multi_chunk | answer | prompt | missing critical facts: purchase order number |
| G16 | development | multi_chunk | answer | chunking | missing critical facts: booking in |
| G19 | development | synonym_heavy | refuse | prompt | missing critical facts: partial delivery |
| G22 | development | synonym_heavy | refuse | prompt | missing critical facts: subcontracting, prior written approval |

Summary:

{
  "prompt": 4,
  "chunking": 1
}

