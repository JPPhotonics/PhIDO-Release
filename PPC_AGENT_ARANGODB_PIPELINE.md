## PPC Agent + ArangoDB Entity Extraction Pipeline (Current Implementation)

This document describes how the **ArangoDB-powered PPC Agent pipeline** currently works end-to-end, including:
- how PDFs become LLM-ready text,
- how entities are extracted and described,
- how ArangoDB is used for vector-based normalization,
- how “Known vs New” decisions are made,
- where the OWL ontology is used today (and where it is *not* used).

---

## High-level architecture

The PPC Agent (`PhotonicsAI/KnowledgeBase/agents/ppc_agent/ppc_agent.py`) runs these stages:

1. **Stage 0 (Preprocess document)**: PDF → structured markdown (preferred) → LLM section isolation → “filtered text”
2. **Phase A (Entity extraction)**: LLM extracts `RawEntity` list (name/type/context), guided by ontology schema
3. **Phase A.5 (Entity describer)**: build richer evidence packs + LLM produces detailed descriptions with evidence quotes/metrics
4. **Phase B (Normalizer)**: resolve acronyms → ArangoDB vector search against KB embeddings → `NormalizedEntity` list
5. **Phase C (Conflict detector)**: decide known/new using thresholds + exact-match override + (light) logic checks + optional LLM judge for ambiguous cases
6. **Return**: JSON with `known_entities` + `new_concepts`

Key output types live in `PhotonicsAI/KnowledgeBase/agents/ppc_agent/models.py`.

---

## Components and where they live

- **Orchestrator**
  - `PhotonicsAI/KnowledgeBase/agents/ppc_agent/ppc_agent.py` (`PPCAgent`)

- **Stage 0: preprocessing**
  - `PhotonicsAI/KnowledgeBase/agents/ppc_agent/llm_document_preprocessor.py` (`LLMDocumentPreprocessor`)
  - `PhotonicsAI/KnowledgeBase/agents/ppc_agent/mcp_client_subprocess.py` (`SubprocessMCPClient`)

- **Phase A: extraction**
  - `PhotonicsAI/KnowledgeBase/agents/ppc_agent/entity_extractor.py` (`EntityExtractor`)

- **Phase A.5: describer**
  - `PhotonicsAI/KnowledgeBase/agents/ppc_agent/entity_describer.py` (`EntityDescriber`)

- **Phase B: normalization / retrieval**
  - `PhotonicsAI/KnowledgeBase/agents/ppc_agent/normalizer.py` (`Normalizer`)
  - `PhotonicsAI/KnowledgeBase/agents/ppc_agent/kb_grounding_tool.py` (LangChain-style tool wrapper)
  - `PhotonicsAI/KnowledgeBase/agents/ppc_agent/acronym_agent.py` (`AcronymAgent`)
  - ArangoDB client + retrieval:
    - `PhotonicsAI/KnowledgeBase/ArangoDB/client.py` (`KnowledgeBaseClient`)
    - `PhotonicsAI/KnowledgeBase/ArangoDB/retrieval.py` (`RetrievalEngine`)

- **Phase C: collision handling**
  - `PhotonicsAI/KnowledgeBase/agents/ppc_agent/conflict_detector.py` (`ConflictDetector`)

- **Ontology helper (T-Box loader)**
  - `PhotonicsAI/KnowledgeBase/agents/ppc_agent/ontology_loader.py` (`OntologyLoader`)
  - Ontology file: `PhotonicsAI/KnowledgeBase/GenerativeOntology/ontology/pic_ontology.ttl`

---

## Stage 0: PDF → “filtered text”

### 0.1 PDF → structured markdown (preferred path)

When `PPCAgent.process_paper(pdf_path=...)` is called, it uses `LLMDocumentPreprocessor.preprocess_pdf()`.

The preferred path is:

1. `SubprocessMCPClient.parse_pdf_to_markdown(pdf_path)`
2. This calls the MCP tool `AxDocumentParser_parse_pdf_to_md`
3. That tool typically returns a message like:
   - `Generated markdown at: /path/to/file.md`
4. `parse_pdf_to_markdown()` detects this and **reads the file contents** to obtain the markdown.

This is implemented in:
- `PhotonicsAI/KnowledgeBase/agents/ppc_agent/mcp_client_subprocess.py`
- `PhotonicsAI/KnowledgeBase/agents/ppc_agent/llm_document_preprocessor.py`

### 0.2 Identify relevant sections (LLM)

Next, the preprocessor runs `_identify_relevant_sections(full_text)`:

- System prompt tells the model to extract excerpts about:
  - architectures/components,
  - specs,
  - design functions,
  - physical principles,
  - properties/metrics.

It returns a JSON list of sections:

```json
[
  {"text": "...", "section_type": "Abstract", "reason": "..."},
  {"text": "...", "section_type": "Results", "reason": "..."}
]
```

Robust JSON parsing is done with a **balanced-bracket extractor** (to avoid regex truncation).

### 0.3 Combine sections

`_combine_sections()` concatenates the returned sections into a single “filtered text” string with markers:

- `[Section i: <type>]`
- `Reason: ...`
- section text

This “filtered text” is the only text passed downstream to entity extraction.

---

## Phase A: ontology-guided entity extraction

### 1.1 Ontology schema injection (T-Box)

`EntityExtractor` loads `pic_ontology.ttl` via `OntologyLoader.get_ontology_schema()`.

Important: today the ontology file defines **high-level classes** (Component, Architecture, Property, …) and properties, not a full taxonomy of all concrete entities.

The loader output is injected into the system prompt as a “Class Hierarchy” hint.

### 1.2 LLM extraction output

The extractor asks the LLM to produce a JSON list of:
- `name` (exact string from text)
- `entity_type` ∈ {Component, Architecture, Property, Design_Function, Physical_Principle}
- optional `context`

Result is parsed into `RawEntity` objects (Pydantic).

---

## Phase A.5: Entity describer (richer descriptions + evidence)

This phase exists because extractor “context” is usually too short for KB ingestion and debugging.

### 2.1 Build a context pack (no LLM)

`EntityDescriber` constructs a **ContextPack** per entity from the filtered text:

- **Section blocks**: full `[Section …]` blocks that contain the entity name
- **Mention windows**: multiple local windows around occurrences (configurable count + window size)
- **Numeric/unit sentences**: sentences containing the entity name and a number/unit token (dB, GHz, nm, V, …)
- **Neighbor entities**: other extracted entities co-mentioned near that entity

This becomes a single string called `context_pack`.

### 2.2 Generate descriptions (LLM)

`EntityDescriber.describe_entities()` makes one batched LLM call to produce:
- `description`: 2–6 sentence grounded technical summary
- `evidence_quotes`: verbatim short quotes
- `key_metrics`: metric statements (value/unit)
- `related_entities`: subset of the neighbor list

These describer artifacts are then passed forward as part of `raw_entities_context` in the orchestrator.

---

## Phase B: Normalization (ArangoDB retrieval)

### 3.1 Acronym resolution

Before searching, `Normalizer` resolves acronyms using `AcronymAgent`:

- Example: `"MZI"` → `"Mach-Zehnder Interferometer"`
- The agent maintains a persistent mapping in `acronym_mappings.json`.
- It is constrained to *only* expand acronyms (not general term normalization).

### 3.2 Vector search (only)

The pipeline currently uses **vector-only search** (keyword/BM25 is removed).

Mechanism:

1. `Normalizer._vector_search(query_name, collection)` calls a KB grounding tool:
   - `kb_tool.invoke({"entity_name": query, "collection": ..., "threshold": 0.3})`
2. Under the hood, this routes to ArangoDB retrieval (`RetrievalEngine.semantic_search` style logic) and returns matches:
   - `[{ "name": "...", "similarity": 0.498, ... }, ...]`
3. The normalizer takes the **top match** and emits `NormalizedEntity`:
   - `raw_name` (paper entity)
   - `kb_name` (best KB match)
   - `similarity` (vector similarity)

### 3.3 What is embedded in ArangoDB?

Embeddings are generated during YAML import by `PhotonicsAI/KnowledgeBase/ArangoDB/importer.py`:

- embedding text = **name + description + equations**

This is why a short query can sometimes yield “moderate” similarity even when the name matches exactly: the stored embedding is influenced by the full description text.

---

## Phase C: “Known vs New” collision handling

The collision logic is implemented in `PhotonicsAI/KnowledgeBase/agents/ppc_agent/conflict_detector.py`.

### 4.1 Exact-match override (name equality)

If `raw_name` equals `kb_name` case-insensitively, similarity is forced to `1.0`.

This protects you from:
- long-description embeddings dragging similarity down, even for exact label matches.

### 4.2 Threshold-based decision

- `HIGH_SIMILARITY = 0.8` → **Known**
- `LOW_SIMILARITY = 0.4` → **New**
- Between 0.4 and 0.8 → **Ambiguous**

### 4.3 Logic validation (lightweight)

`_logic_validate()` runs lightweight heuristic checks using ontology constraints (if present).
It does **not** do full OWL reasoning today.

If logic flags are present, entity is treated as “new/hallucinated” and labeled:
- `description = "Flagged: <reasons>"`

### 4.4 Ambiguous judge (LLM)

If similarity is ambiguous (0.4–0.8), a judge LLM prompt decides `"known"` vs `"new"` using:
- the raw name,
- KB match name,
- similarity,
- and the evidence context (now richer due to the describer).

### 4.5 NewConcept payload (now richer)

For new concepts, `NewConcept` now includes:
- `description` (LLM describer summary; fallback to evidence text)
- `context` (evidence pack)
- `evidence_quotes`
- `key_metrics`
- `related_entities`

This is useful both for:
- human review,
- future KB ingestion steps.

---

## Where the ontology is used vs not used

### Used today
- Injected into **entity extraction prompts**
- Used for lightweight logic checks in **conflict detection**

### Not used today
- PPC Agent does **not** use the `arango-rdf` / RDF export / SHACL validation path at runtime.
- The RDF export/validation is currently separate in:
  - `PhotonicsAI/KnowledgeBase/GenerativeOntology/src/graph_manager.py`
  - exercised by `arangoDB_test.py`

---

## Where to tweak behavior

### LLM model choices
- `PPCAgent(llm_model=...)` controls:
  - preprocessing section isolation,
  - entity extraction,
  - entity describer,
  - ambiguous-judge.

### Retrieval thresholds
- Vector filter threshold in normalizer:
  - `Normalizer._vector_search(... threshold=0.3)`
- Known/new thresholds:
  - `ConflictDetector.HIGH_SIMILARITY` / `LOW_SIMILARITY`

### Context pack size
In `EntityDescriber.describe_entities(...)`:
- `max_mentions`
- `mention_window_sentences`
- `max_section_blocks`
- `ContextPack.render(max_chars=...)`

---

## Outputs and debugging

- Detailed JSON reports:
  - `ppc_agent_detailed_report_pdf.json`
  - `ppc_agent_detailed_report_text.json`
  - Include grouped LLM calls by stage:
    - preprocessing
    - entity_extraction
    - entity_description
    - conflict_handling

These are produced by `LLMTracer` in `test_ppc_agent.py`.


