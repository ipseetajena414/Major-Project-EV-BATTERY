# Architecture (Member 5 integration)

```mermaid
flowchart TD
  M1[Member 1: master_table.parquet - read only] --> M2[Member 2: Member2Predictor + health contract]
  M2 -- observed SOH % default / predicted SOH % experimental opt-in --> A[m5integration adapters]
  A --> M3[Member 3: RangeCalculator.estimate]
  A --> M4[Member 4: bms.py rules + LangGraph simulation]
  M3 -. vendored copy used internally .-> M4
  UI[Streamlit app.py + dashboard/] --> A
  RAG[rag/: loader, chunker, TF-IDF store, QA] --> UI
```

## Data flow
```mermaid
flowchart LR
  P[PDF upload + user-entered metadata] --> L[document_loader] --> C[text_chunker] --> V[vector_store: TF-IDF, persisted in data/rag_index] --> R[retriever] --> Q[qa_engine]
  Q --> O[verbatim excerpts + citations; optional separate LLM interpretation]
```

## Multi-agent interaction (Member 4)
```mermaid
flowchart LR
  D[Driver Persona: wants fast charge to 100%] --> B[BMS Supervisor: deterministic Python rules in bms.py]
  B --> X[apply action] --> S[update battery state - placeholder degradation] --> E{end of life?}
  E -- no --> D
  E -- yes --> OUT[results]
  LLM[optional Gemini] -. explanation text only .-> B
```

## Boundaries enforced in code
- `member2_adapter.soh_for_range`: only `ok`/`ok_experimental` Member 2 results can feed range; observed SOH is the default; predicted SOH is labelled experimental; blocked results raise `RangeBlocked`.
- `member2_adapter.rul_signal` always returns `None` (RUL not approved).
- Member 3 is called with SOH in percent and speed already in m/s (cycle CSVs); no unit conversion needed.
- Safety decisions come only from `ev_member4/bms.py`; the LLM never changes them.
- Cell-level Ah is never converted to pack kWh; pack energy comes from the vehicle JSON.
