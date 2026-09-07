# Day 7 failure classification

Every failed case has exactly one primary type: chunking, retrieval, prompt, citation, refusal, or evaluator.

| Case ID | Split | Category | Observed failure | Primary type | Reason |
|---|---|---|---|---|---|
| G13 | train | multi_chunk | answer | prompt | missing critical facts: sixty days |
| G14 | development | multi_chunk | answer | prompt | missing critical facts: purchase order number |
| G16 | development | multi_chunk | answer | chunking | missing critical facts: booking in |
| G18 | train | synonym_heavy | answer | chunking | missing critical facts: assign or novate |
| G19 | development | synonym_heavy | refuse | prompt | missing critical facts: partial delivery |
| G21 | train | synonym_heavy | answer | chunking | missing critical facts: thirty days |
| G22 | development | synonym_heavy | refuse | prompt | missing critical facts: subcontracting, prior written approval |
| G23 | train | unanswerable | answer | refusal | observed answer, expected refuse |

Summary:

{
  "prompt": 4,
  "chunking": 3,
  "refusal": 1
}

