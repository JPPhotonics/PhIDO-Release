# PPC Agent (Pre-processing & Context Agent)

The PPC Agent is the first stage of a multi-agent system for ingesting AMF photonics papers into the ArangoDB knowledge base. It processes raw paper text/PDFs to extract and normalize entities, identifying both known entities (that match existing KB entries) and new concepts (that need to be added).

## Overview

The PPC Agent uses a multi-phase approach:

1. **Stage 0: Text Pre-filter** (Non-LLM) - Quickly filters paper text to isolate relevant sections
2. **Phase A: Raw Entity Extraction** (LLM) - Extracts candidate entities from filtered text
3. **Phase B: Context Retrieval & Normalization** (LLM + Tool) - Normalizes entities against KB using semantic search
4. **Phase C: Conflict & Novelty Identification** (LLM) - Categorizes entities into known vs new
5. **Phase D: Structured Output** - Generates JSON payload for downstream agents

## Architecture

```
┌─────────────────┐
│  PDF/Text Input │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stage 0: Filter │  (Non-LLM: Section isolation + keyword filtering)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Phase A: Extract│  (LLM: Zero-Shot CoT entity extraction)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Phase B: Normalize│ (LLM + KB_Grounding_Tool: Semantic search)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Phase C: Categorize│ (LLM: Known vs New classification)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  JSON Output     │  (known_entities + new_concepts)
└─────────────────┘
```

## Installation

The PPC Agent requires:

```bash
pip install PyPDF2 langchain langchain-openai langchain-community
```

### MCP Integration (Optional but Recommended)

The PPC Agent supports using the AxDocumentAnnotator MCP tool for intelligent PDF annotation.
This replaces the previous text filtering approach with AI-powered document analysis.

**To enable MCP annotation:**

1. **Install MCP Python SDK**:
   ```bash
   pip install mcp
   ```

2. **Ensure the `axiomatic-mcp` server is configured** in Cursor (see `.cursor/mcp.json`)

3. **The MCP tool will be automatically used** when processing PDFs

4. **If MCP annotation fails**, the agent falls back to direct PDF text extraction (PyPDF2)

**The MCP annotator provides:**
- Intelligent section extraction focused on photonic components and architectures
- Context-aware annotation with page references
- Better handling of technical content compared to keyword-based filtering

**Note:** The MCP client connects to the server using stdio (standard input/output), matching how Cursor runs MCP servers. See `MCP_SETUP.md` for detailed setup instructions.

## Usage

### Basic Usage

```python
from pathlib import Path
from PhotonicsAI.KnowledgeBase.agents.ppc_agent import PPCAgent
from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient

# Initialize KB client
kb_client = Neo4jClient()
kb_client.connect()

# Create PPC Agent
agent = PPCAgent(kb_client=kb_client, llm_model="gemini-2.5-pro")

# Process a PDF
result = agent.process_paper(pdf_path=Path("paper.pdf"))

# Or process raw text
result = agent.process_paper(text="... paper text ...")

# Result is a dict with 'known_entities' and 'new_concepts'
print(f"Found {len(result['known_entities'])} known entities")
print(f"Found {len(result['new_concepts'])} new concepts")
```

### Output Format

The agent returns a JSON-serializable dictionary:

```python
{
    "known_entities": [
        {
            "raw_name": "MZI",
            "kb_name": "Mach–Zehnder Interferometer",
            "similarity": 0.95,
            "entity_type": "Architecture",
            "collection": "Architectures"
        },
        ...
    ],
    "new_concepts": [
        {
            "name": "Novel Ring Modulator",
            "entity_type": "Architecture",
            "description": "A new ring modulator design...",
            "context": "proposed in section 3"
        },
        ...
    ]
}
```

### Alternative Output Methods

```python
# Get JSON string
json_str = agent.process_paper_to_json(pdf_path=Path("paper.pdf"))

# Get PPCResult object
result_obj = agent.process_paper_to_result(pdf_path=Path("paper.pdf"))
for entity in result_obj.known_entities:
    print(f"{entity.raw_name} -> {entity.kb_name} (similarity: {entity.similarity})")
```

## Components

### Stage 0: Document Annotation (`mcp_document_annotator.py`)

**NEW**: MCP-based document annotation using AxDocumentAnnotator:
- **Intelligent Extraction**: Uses AI to extract relevant sections about photonic components and architectures
- **Context-Aware**: Provides detailed annotations with page references and contextual descriptions
- **Query-Based**: Uses a specialized query to focus on:
  - Photonic components (modulators, detectors, waveguides, etc.)
  - Photonic architectures (interferometers, ring resonators, etc.)
  - Component properties (insertion loss, bandwidth, Q factor, etc.)
  - Design functions and physical principles

**Legacy**: Text Filter (`text_filter.py`) - Still available as fallback:
- **Section Isolation**: Identifies and extracts relevant sections (Abstract, Introduction, Proposed Architecture, Results, Conclusion)
- **Keyword Filtering**: Extracts paragraphs containing:
  - Architecture/Component names (capitalized, hyphenated terms)
  - Key action verbs (demonstrate, propose, achieve, implement, novel)
  - High-value properties (insertion loss, bandwidth, Q factor, efficiency)

### Phase A: Entity Extractor (`entity_extractor.py`)

LLM-based entity extraction using Zero-Shot Chain-of-Thought reasoning:
- Extracts candidates for: Components, Architectures, Properties, Design_Functions, Physical_Principles
- Uses structured output (Pydantic models) for reliable parsing
- Includes fallback JSON parsing for robustness

### Phase B: Normalizer (`normalizer.py`)

Normalizes entities against the knowledge base:
- Uses `KB_Grounding_Tool` to perform semantic search
- Maps entity types to KB collections
- Returns similarity scores for matching

### Phase C: Conflict Detector (`conflict_detector.py`)

Categorizes entities into known vs new:
- **High similarity (≥0.8)**: Known entity
- **Low similarity (<0.5)**: New concept
- **Ambiguous (0.5-0.8)**: Uses LLM to decide

### KB Grounding Tool (`kb_grounding_tool.py`)

LangChain tool wrapper for semantic search:
- Read-only tool (safety)
- Returns JSON with top matches and similarity scores
- Used for normalization and conflict detection

## Configuration

### LLM Model Selection

The agent supports any model available in `llm_api.py`:

```python
agent = PPCAgent(
    kb_client=kb_client,
    llm_model="gemini-2.5-pro"  # or "gpt-4o", "claude-opus-4-20250514", etc.
)
```

### Similarity Thresholds

Thresholds are configurable in `conflict_detector.py`:
- `HIGH_SIMILARITY = 0.8` - Known entity threshold
- `LOW_SIMILARITY = 0.5` - New concept threshold
- Ambiguous range: 0.5-0.8 (uses LLM to resolve)

## Integration with Downstream Agents

The PPC Agent output is designed to be consumed by:

1. **VSA Agent (Validation and Synthesis Agent)**: Validates relationships and formats payload
2. **DIA Agent (Database Integration Agent)**: Executes ACID transactions to add new entities

Example handoff:

```python
# PPC Agent output
ppc_result = agent.process_paper(pdf_path=Path("paper.pdf"))

# Pass to VSA Agent
vsa_agent.validate_and_synthesize(ppc_result)
```

## Error Handling

The agent includes robust error handling:
- PDF parsing errors: Falls back to text extraction
- LLM extraction failures: Uses JSON parsing fallback
- KB connection errors: Handles gracefully with empty results
- Ambiguous entities: Defaults to "new concept" (safer)

## Performance Considerations

- **Stage 0 filtering**: Reduces LLM input by 60-80%, saving tokens and cost
- **Batch processing**: Can process multiple papers sequentially
- **Caching**: KB embeddings are cached (no regeneration needed)

## Troubleshooting

### No entities extracted

- Check if filtered text is empty (may need to adjust filter criteria)
- Verify LLM API keys are set
- Check entity extraction prompt for issues

### Low similarity scores

- Verify KB has embeddings (run import if needed)
- Check if entity names are too generic
- Consider lowering threshold for broader matching

### PDF extraction fails

- Ensure PyPDF2 is installed: `pip install PyPDF2`
- Some PDFs may have encoding issues (agent handles gracefully)

## Future Enhancements

- [ ] Support for figure/image analysis
- [ ] Multi-document batch processing
- [ ] Confidence scores for extracted entities
- [ ] Relationship extraction (not just entities)
- [ ] Incremental learning from corrections

## License

Part of the PhIDO project.

