"""
Interpreter Agent — agentic tool-calling loop that produces a DesignIntent.

The LLM autonomously decides which tools to call and when to stop.
Two tool categories serve different roles:
  - PDK tools: authoritative source for fabrication details (port configs, modules, args)
  - KG tools:  foundational domain knowledge (what a concept IS, how it works, sub-components)

The agent consults the KG when it encounters concepts it doesn't understand or is
uncertain about — not as a mandatory step for every component.

After free exploration, a **KG Grounding Gate** checks whether the agent made any KG
calls at all. If it never consulted the KG (zero calls), it prompts a self-audit:
"Are you confident about all these concepts, or are you making assumptions?"
If the agent did consult the KG at least once, we trust its judgment.

Phases:
  Phase 0    — LLM concept extraction from user prompt
  Phase 1    — Free agentic exploration (tool calls driven by LLM)
  Phase 1.5  — KG Grounding Gate (self-audit prompt if zero KG calls were made)
  Phase 1.75 — Disambiguation (agent asks user clarification questions)
  Phase 2    — Structured output extraction (DesignIntent)
  Phase 3    — Critic review (independent agent verifies output; loops back on failure)

The pipeline is split into two entry points for Streamlit compatibility:
  explore_and_ask()  — Phases 0 → 1 → 1.5 → 1.75 (pauses for user input)
  finalize_stream()  — Phases 2 → 3 (resumes with optional clarifications)

Usage:
    from mcp_servers.interpreter_agent import interpret
    result = interpret("A 1x2 beamsplitter connected to two MZI modulators with heaters")
    print(result.summary())
"""

import json
import os
from typing import Any, Generator, Optional

from pydantic import BaseModel, Field
from openai import OpenAI
from mcp_servers.models import (
    DesignIntent, ComponentIntent, Connection,
    ClarificationQuestion, ClarificationRequest,
)


# ---------------------------------------------------------------------------
# Concept extraction model (for the LLM pre-pass)
# ---------------------------------------------------------------------------

class ExtractedConcepts(BaseModel):
    """Photonic domain concepts extracted from a user prompt."""
    components: list[str] = Field(
        ..., description="Distinct photonic component types, architectures, "
        "or design functions (e.g. 'MZI', 'ring resonator', 'Clements scheme', "
        "'wavelength demultiplexing')")
    parameters: list[str] = Field(
        ..., description="Physical parameters or properties mentioned "
        "(e.g. 'extinction ratio', 'arm length', 'insertion loss', 'FSR', 'bandwidth')")
    specs: list[str] = Field(
        ..., description="Specific measurements or numeric values with units "
        "(e.g. '1550 nm', '10 dB', '150 um', '40 GHz bandwidth')")


# ---------------------------------------------------------------------------
# Critic verdict model (for the Phase 3 critic loop)
# ---------------------------------------------------------------------------

class CriticIssue(BaseModel):
    """A single issue found by the critic agent."""
    severity: str = Field(
        ..., description="'major' (requires re-exploration with tools) or 'minor' (fixable in structuring)")
    description: str = Field(..., description="What is wrong and why")
    affected_components: list[str] = Field(
        default_factory=list,
        description="Component IDs affected (e.g. ['C1', 'C3']) or ['global'] for cross-cutting issues")


class CriticVerdict(BaseModel):
    """Structured verdict from the critic agent."""
    passed: bool = Field(..., description="True if the DesignIntent faithfully captures the user's request")
    issues: list[CriticIssue] = Field(default_factory=list, description="Issues found (empty if passed)")
    summary: str = Field(..., description="Brief explanation of the verdict")

# ---------------------------------------------------------------------------
# Direct tool imports
# ---------------------------------------------------------------------------
from mcp_servers.pdk_catalog_server import (
    search_components,
    validate_port_config,
    get_component_details,
    get_module_params,
)

try:
    from mcp_servers.kg_server import (
        search_concepts,
        resolve_function,
        get_concept_neighborhood,
        get_component_properties,
    )
    _KG_AVAILABLE = True
except Exception as _kg_err:
    _KG_AVAILABLE = False
    print(f"KG tools unavailable (non-fatal): {_kg_err}")

# ---------------------------------------------------------------------------
# Tool registry: OpenAI function-calling specs + dispatch table
# ---------------------------------------------------------------------------

# -- PDK tools (always available) ------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_pdk",
            "description": (
                "Search the PDK component catalog for photonic components matching "
                "a natural-language description. Returns top 5 matches with module names, "
                "port configs, labels, and match scores."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Component description, e.g. 'MZI modulator with heater' or '1x2 beamsplitter'",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_ports",
            "description": (
                "Check whether a port configuration (e.g. '2x2', '1x2') is valid "
                "for a given component type in the PDK. Returns available port configs if invalid."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "component_query": {"type": "string", "description": "Component name or keyword, e.g. 'mmi' or 'MZI'"},
                    "port_config": {"type": "string", "description": "Port config to validate, e.g. '2x2'"},
                },
                "required": ["component_query", "port_config"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_component_info",
            "description": (
                "Get full details for a specific PDK component by its exact module name. "
                "Returns ports, args, technology, bandwidth, full docstring. "
                "Use after search_pdk to inspect a specific match."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "module_name": {
                        "type": "string",
                        "description": "Exact module name from PDK, e.g. 'mzi_2x2_heater_tin_cband' or '_mmi1x2'",
                    }
                },
                "required": ["module_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_module_params",
            "description": (
                "Get the default GDSFactory settings/parameters for a specific PDK component. "
                "Returns all parameter names and their default values. Use after search_pdk "
                "to understand what parameters a component accepts (e.g., arm_length, gap, "
                "coupling_length) and their defaults."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "module_name": {
                        "type": "string",
                        "description": "Exact PDK module name, e.g. 'mzi_2x2_heater_tin_cband'",
                    }
                },
                "required": ["module_name"],
            },
        },
    },
]

# Dispatch table: tool name -> callable
_TOOL_DISPATCH = {
    "search_pdk": lambda args: search_components(args["query"]),
    "validate_ports": lambda args: validate_port_config(args["component_query"], args["port_config"]),
    "get_component_info": lambda args: get_component_details(args["module_name"]),
    "get_module_params": lambda args: get_module_params(args["module_name"]),
}

# -- KG tools (require Neo4j) -----------------------------------------------
_KG_TOOL_NAMES = frozenset()  # populated below if KG is available

if _KG_AVAILABLE:
    _kg_tool_defs = [
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_graph",
                "description": (
                    "Semantic search across the photonics knowledge graph for foundational "
                    "domain knowledge. Finds concepts (components, architectures, properties, "
                    "physical principles) matching a query. Use when you are unsure what a "
                    "photonic concept is or how it works. Does NOT contain fabrication-specific "
                    "details like port configs — use PDK tools for that."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Concept to search for, e.g. 'ring resonator modulator'"},
                        "entity_type": {
                            "type": "string",
                            "description": "Optional filter: Components, Architectures, Properties, Design_Functions, or Physical_Principles",
                            "default": "",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "resolve_function",
                "description": (
                    "Reverse lookup: given a design function (e.g. 'modulation', 'wavelength filtering'), "
                    "find which components or architectures can perform it."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "function_name": {"type": "string", "description": "Design function, e.g. 'Modulation', 'Wavelength_Filter'"},
                    },
                    "required": ["function_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_concept_neighborhood",
                "description": (
                    "Get a concept's graph neighborhood: its relationships, sub-components, "
                    "properties, and connected entities. Use to understand what an architecture "
                    "is made of or what principles a component uses."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "concept_name": {"type": "string", "description": "Concept name, e.g. 'MZI' or 'Ring_Resonator'"},
                    },
                    "required": ["concept_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_component_properties",
                "description": (
                    "Get foundational knowledge about a component or architecture from the "
                    "knowledge graph: what properties it has, what design functions it performs, "
                    "what physical principles it relies on, and what sub-components make it up. "
                    "Use when you need to understand WHAT something is, not fabrication specifics."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "component_name": {
                            "type": "string",
                            "description": "Component name, e.g. 'MZI', 'Ring_Resonator', 'Directional_Coupler'",
                        },
                    },
                    "required": ["component_name"],
                },
            },
        },
    ]

    TOOLS.extend(_kg_tool_defs)

    _TOOL_DISPATCH.update({
        "search_knowledge_graph": lambda args: search_concepts(args["query"], args.get("entity_type", "")),
        "resolve_function": lambda args: resolve_function(args["function_name"]),
        "get_concept_neighborhood": lambda args: get_concept_neighborhood(args["concept_name"]),
        "get_component_properties": lambda args: get_component_properties(args["component_name"]),
    })

    _KG_TOOL_NAMES = frozenset([
        "search_knowledge_graph",
        "resolve_function",
        "get_concept_neighborhood",
        "get_component_properties",
    ])


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """\
You are a photonic circuit interpreter. Your task is to understand a user's natural-language 
description of a photonic circuit and extract a structured DesignIntent.

FOUNDATIONAL PRINCIPLE: Your pretrained knowledge may be outdated, incomplete, or incorrect 
for this specific PDK and domain. The PDK and Knowledge Graph are the ground truth. You MUST 
verify every claim through tools. Do not assume you know what a component is, how it works, 
or what port configurations it has — look it up.

You have access to two categories of tools:

**PDK tools** — the authoritative source for what can be fabricated:
  - search_pdk: find components by keyword. Returns module names, port configs, labels.
  - validate_ports: verify a specific port configuration against the PDK.
  - get_component_info: full technical details for a PDK module (ports, args, technology).
  Every component in your output must be backed by a PDK search. Port configs, module names,
  and fabrication parameters come ONLY from the PDK — never from your own knowledge.

**Knowledge Graph (KG) tools** — the authoritative source for domain knowledge:
  - search_knowledge_graph: semantic search for photonic concepts (components, architectures,
    properties, physical principles).
  - resolve_function: given a design function (e.g. "modulation"), find which component types
    or architectures can perform it.
  - get_concept_neighborhood: explore a concept's relationships — what sub-components it uses,
    what physical principles it relies on, what properties it has.
  - get_component_properties: get all known properties, design functions, physical principles,
    and sub-components for a concept.
  The KG defines what a component IS, how it works, and what it's made of. Consult the KG 
  for every component and concept — do not rely on your training data for domain knowledge.

WORKFLOW:
1. Read the user's prompt. Identify every distinct component, connection, and specification.
2. For EVERY component type mentioned or implied:
   a. Search the PDK to check if a matching fabrication component exists and confirm port
      configs and available parameters.
   b. Consult the KG to verify your understanding of the component — what it is, how it 
      works, what sub-components or principles it involves. Use get_component_properties 
      or search_knowledge_graph for this. Do this for every component, not just unfamiliar
      ones. Examples:
        - User mentions "VOA" → check KG for typical implementations.
        - User says "arrayed waveguide grating" → check KG for its architecture.
        - User mentions a function like "wavelength demultiplexing" → use resolve_function
          to discover which components can perform it.
3. ARCHITECTURAL DECOMPOSITION — only when needed:
   a. For each component the user described, FIRST check whether the PDK has a monolithic
      component that directly satisfies the full description. Search with specific terms
      that capture the complete device (e.g. "MZI with heater", "ring resonator modulator").
      If a good PDK match exists, USE IT — this is always preferred.
   b. Only if no suitable monolithic PDK component exists, decompose the user's described
      component into sub-components. Use get_concept_neighborhood or get_component_properties 
      from the KG to discover the correct decomposition — do NOT guess the sub-component 
      breakdown from general knowledge.
   c. After decomposition, search the PDK for EACH sub-component to verify availability and
      get port configs.
   d. The goal is a FABRICABLE design: every component in the final output must either be a
      monolithic PDK component or be decomposed into PDK-available parts. Anything else
      must be flagged as an ambiguity.
   e. When decomposing, preserve the user's intent — if the user said "MZI", the title/summary
      should still reference "MZI" but the components list should contain the actual sub-
      components that form it, with connections showing the internal topology.
4. If anything is ambiguous — implementation choice, connection topology, missing specs —
   note it explicitly as an ambiguity. If there are multiple valid decompositions, note
   which one you chose and why.

CRITICAL RULES:
- Port configurations come ONLY from the PDK. Do NOT guess port configs — verify with
  validate_ports or get_component_info.
- Domain knowledge comes from the KG. Do NOT rely on your training data for how components
  work, what they're made of, or how to decompose them — verify with KG tools.
- Any claim about a component that is not backed by a tool result from this conversation
  must be flagged as an ambiguity.
- Each physical instance gets its own id (C1, C2, ...). "Two modulators" → C1 and C2.
- When decomposing a composite architecture, each sub-component gets its own id and the
  connections between them must reflect the internal topology.
- Track confidence per component:
    1.0 = PDK match confirmed AND KG-verified understanding
    0.7 = PDK match confirmed but KG not consulted, OR KG-verified but weak/no PDK match
    0.4 = neither PDK nor KG verification performed

When you are satisfied, say DONE and summarize your findings including what you verified 
(citing tool results), any decompositions you performed (citing KG results), and any 
remaining ambiguities. Do NOT output the DesignIntent JSON yourself.
"""

GROUNDING_GATE_PROMPT = """\
GROUNDING REQUIRED: You mentioned the following concepts but did not consult the Knowledge 
Graph for any of them:

{ungrounded_items}

Every component and concept must be verified against the KG — your training data is not a 
reliable substitute. Use get_component_properties or get_concept_neighborhood to look up 
each of the above concepts now. This is not optional.
"""

STRUCTURING_PROMPT = """\
Based on the preceding conversation — the user's original prompt, all PDK search results,
and all Knowledge Graph results — produce the final DesignIntent.

IMPORTANT: Only include information that is backed by tool results from this conversation.
Any claim about a component, port config, or architecture that was NOT verified by a PDK 
or KG tool call must be flagged as an ambiguity.

Rules:
- Each physical instance gets a unique id (C1, C2, ...).
- port_config: only set if confirmed by PDK tools (validate_ports or get_component_info).
  NEVER guess a port config — this must come from the PDK.
- specs: only physical parameters from the original user prompt (arm_length, bandwidth, etc.).
  Do NOT put PDK module names or match scores in specs.
- role: functional purpose (modulator, splitter, detector, filter, coupler, etc.).
- connections: use component ids + natural language description.
- When a user-described component was decomposed into sub-components (e.g. an MZI decomposed
  into splitter + arms + combiner), each sub-component is a separate entry with its own id.
  The connections must reflect the internal topology of the decomposed architecture, not just
  "MZI connected to diode". The description field should note which higher-level architecture
  the sub-component belongs to (e.g. "1x2 MMI splitter — input coupler of MZI").
- confidence scoring:
    1.0 = PDK match confirmed AND understanding verified by KG
    0.7 = PDK match confirmed but KG not consulted, OR KG-verified but weak/no PDK match
    0.4 = neither PDK nor KG verification — flag as ambiguity
- source_span: verbatim substring from the user's original input.
- ambiguities: list anything you assumed or couldn't verify through tools.
  If the KG revealed multiple possible implementations, note that as an ambiguity.
  If a component description relies on your own knowledge rather than tool results, note it.
"""

CRITIC_SYSTEM_PROMPT = """\
You are a rigorous critic reviewing a photonic circuit DesignIntent that was extracted from 
a user's natural-language prompt. Your job is to verify that the DesignIntent **faithfully 
and completely** captures what the user asked for, and that claims are grounded in tool 
results rather than assumptions.

You have access to the same PDK and Knowledge Graph tools as the interpreter. Use them to 
independently verify claims — do NOT trust the interpreter's work blindly. Your tool calls 
are the source of truth.

You will be given:
1. The original user prompt
2. Extracted concepts (components, parameters, specs) from the prompt
3. (If applicable) User clarifications — answers the user gave to the interpreter's 
   disambiguation questions. These are authoritative extensions of the original prompt.
4. The DesignIntent JSON produced by the interpreter

Check the following:

**Completeness**
- Every component type from the extracted concepts list appears in the DesignIntent.
- Every parameter/spec mentioned by the user is captured (either in component specs or noted 
  as an ambiguity).
- All connections described by the user are present.
- The correct number of physical instances is created (e.g. "two modulators" → 2 entries).

**Faithfulness**
- No hallucinated components or connections that the user did not ask for.
- Component roles match what the user described.
- No invented specifications — only user-stated values should appear in specs.
- If user clarifications are provided, details derived from those answers are NOT 
  hallucinations. They are valid design requirements even if not in the original prompt.

**PDK accuracy**
- Port configs are only set when they can be verified against the PDK. Use validate_ports 
  or get_component_info to independently check every port_config in the DesignIntent.
- If a port_config is set, verify it exists in the PDK for that component type.
- Every component should have been searched in the PDK. If a component description seems
  to reference a PDK module, verify the module actually exists.

**Tool grounding**
- For each component in the DesignIntent, check whether its description and properties are 
  consistent with what the PDK and KG actually report. Use search_pdk and 
  get_component_properties to spot-check at least the most critical components.
- If a component has confidence 1.0, verify that the claim is justified — it should have 
  both a PDK match and KG-verified understanding. If you cannot confirm this, flag it.
- Any component whose description appears to rely on general knowledge rather than tool 
  results should be flagged as a minor issue ("ungrounded claim").

**Connection topology**
- Connections make physical sense for the described circuit.
- No orphaned components (every component should be connected unless the user described it 
  as standalone).

**Specs integrity**
- User-stated specifications (wavelength, bandwidth, arm length, etc.) are correctly assigned 
  to the right components.
- No specs are duplicated or assigned to wrong components.

**Architectural decomposition**
- If the user described a composite architecture (e.g. "MZI with diodes in each arm"),
  check that the interpreter handled it in ONE of these two valid ways:
    1. Found a monolithic PDK component that directly satisfies the full description
       (verified via search_pdk or get_component_info). This is the PREFERRED outcome
       when available — do NOT penalize the interpreter for using a single component
       that genuinely matches.
    2. Properly decomposed it into fabricable sub-components with correct internal
       topology. Use get_concept_neighborhood to independently verify that the 
       decomposition is architecturally correct — do not trust the interpreter's 
       decomposition at face value.
- Only flag a decomposition issue if the interpreter did NEITHER: i.e. it listed
  a high-level component name that has no direct PDK match AND did not break it down
  into buildable sub-components.
- The connections between sub-components must reflect the actual internal topology,
  not just surface-level "A connected to B".

WORKFLOW:
1. Read the user prompt, extracted concepts, and any user clarifications carefully.
2. Compare against the DesignIntent systematically.
3. Independently verify using your tools — check at least: every port_config, every 
   decomposition, and any component with confidence >= 0.7.
4. Produce your verdict.

For each issue found, classify severity:
- **major**: Missing component, wrong topology, hallucinated element, incorrect port config, 
  unverified decomposition. These require the interpreter to re-explore with tools.
- **minor**: Slightly imprecise description, missing an ambiguity note, minor spec placement,
  ungrounded but plausible claim. These can be fixed in re-structuring.

Be thorough but fair. If the DesignIntent is reasonable, well-grounded in tool results, and 
captures the user's intent correctly, pass it. Do not nitpick stylistic choices.

When you are done analyzing, say VERDICT and state your conclusion. Do NOT output the 
CriticVerdict JSON yourself.
"""

CRITIC_FEEDBACK_PROMPT = """\
CRITIC FEEDBACK (attempt {attempt}/{max_attempts}): The critic agent reviewed your 
DesignIntent and found the following issues:

{issues_text}

Critic summary: {summary}

Please re-investigate using your tools and address these problems. Focus on the major 
issues first. When done, say DONE with your updated findings.
"""

DISAMBIGUATION_PROMPT = """\
Before we finalize the design, review your findings and identify any remaining 
uncertainties, choices, or ambiguities that the user should clarify.

For each question:
- **question**: What you need the user to tell you.
- **context**: What you found in your search that raised this question. Be specific — 
  reference tool results (e.g. "PDK has both mzi_2x2_heater_tin_cband and 
  mzi_2x2_pindiode_cband — which tuning mechanism do you want?").
- **options**: Concrete choices if applicable (e.g. ["TiN heater", "PIN diode"]).
  Leave empty for open-ended questions.
- **default**: What you would assume if the user doesn't answer.
- **priority**: "critical" if the design cannot proceed without an answer (e.g. a 
  fundamental topology choice), or "helpful" if you can make a reasonable default 
  but the user might want to override it.

If your tool results leave no open questions and all components are well-characterized, 
set ready_to_proceed to True with an empty questions list. Do not invent questions just 
to seem thorough — only ask if there is genuine uncertainty informed by your tool results.
"""

CLARIFICATION_INJECTION_PROMPT = """\
The user has provided the following clarifications:

{clarification_items}

Update your analysis to incorporate these answers. If any clarification changes which 
PDK component you should use or how the circuit should be structured, verify with the 
appropriate tools now. When done, say DONE with your updated findings.
"""

UPSTREAM_FEEDBACK_PROMPT = """\
DOWNSTREAM VALIDATION FEEDBACK
===============================
A later stage in the pipeline attempted to build your DesignIntent into a
fabricable circuit and encountered problems. These are NOT hypothetical — they
are concrete failures from PDK validation and schematic construction.

{feedback_items}

You MUST address these problems using your tools. They indicate that your
DesignIntent needs revision — either different components, different decomposition,
or different connections. Re-investigate with search_pdk and the knowledge graph,
then say DONE when you have enough information to produce a corrected DesignIntent.
"""


# ---------------------------------------------------------------------------
# Tool execution + tracking
# ---------------------------------------------------------------------------

def _execute_tool_raw(name: str, arguments: str) -> tuple[dict, str]:
    """Execute a tool call and return (parsed_args, result_string)."""
    args = json.loads(arguments)
    fn = _TOOL_DISPATCH.get(name)
    if fn is None:
        return args, json.dumps({"error": f"Unknown tool: {name}"})
    result = fn(args)
    return args, result


_CONCEPT_EXTRACTION_PROMPT = """\
Extract all domain-specific photonic concepts from the following circuit description.
Categorize into three groups:

1. **components** — device types, architectures, design patterns, circuit topologies
   (e.g. 'MZI', 'ring resonator', 'Clements scheme', 'reconfigurable mesh',
   'wavelength demultiplexing', 'VOA', 'directional coupler')
2. **parameters** — physical properties, performance metrics, design variables
   (e.g. 'extinction ratio', 'arm length', 'insertion loss', 'FSR', 'bandwidth',
   'coupling coefficient', 'thermo-optic phase shift')
3. **specs** — concrete measurements or numeric values with units
   (e.g. '1550 nm', '10 dB', '150 um', '40 GHz bandwidth', '4x4')

Rules:
- Only include domain-specific terms. Exclude generic words like 'circuit',
  'connection', 'design', 'input', 'output', 'signal'.
- Each entry should be a concise term or value, not a sentence.
- If a term could be both a component and a parameter, prefer the more specific category.
"""


def _extract_concepts_llm(
    client: OpenAI, user_prompt: str, model: str
) -> ExtractedConcepts:
    """Use an LLM structured-output call to extract photonic concepts from the user prompt."""
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": _CONCEPT_EXTRACTION_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=ExtractedConcepts,
    )
    msg = response.choices[0].message
    if msg.parsed:
        return msg.parsed
    # Fallback: empty extraction if the model refuses
    return ExtractedConcepts(components=[], parameters=[], specs=[])


# Event type alias for streaming
AgentEvent = dict[str, Any]


def _run_critic(
    client: OpenAI,
    user_prompt: str,
    extracted: ExtractedConcepts,
    design_intent: DesignIntent,
    model: str,
    max_rounds: int = 10,
    clarifications: Optional[dict[str, str]] = None,
) -> Generator[AgentEvent, None, None]:
    """Run the critic agent. Yields streaming events and ends with a 'critic' event.

    The critic has its own separate conversation (not the interpreter's messages)
    and full tool access for independent verification.
    """
    di_json = json.dumps(design_intent.full(), indent=2, ensure_ascii=False)
    concepts_summary = (
        f"Components/architectures: {', '.join(extracted.components) or 'none'}\n"
        f"Parameters/properties: {', '.join(extracted.parameters) or 'none'}\n"
        f"Specs/measurements: {', '.join(extracted.specs) or 'none'}"
    )

    clarification_section = ""
    if clarifications:
        items = "\n".join(f"  - **Q:** {q}\n    **A:** {a}" for q, a in clarifications.items())
        clarification_section = (
            f"\n\n## User Clarifications\n"
            f"After the initial exploration, the agent asked the user clarification "
            f"questions. The user provided these answers, which are now part of the "
            f"design requirements:\n\n{items}\n\n"
            f"Treat these answers as authoritative extensions of the original prompt. "
            f"Details that come from user clarifications are NOT hallucinations."
        )

    critic_user_msg = (
        f"## Original User Prompt\n{user_prompt}\n\n"
        f"## Extracted Concepts\n{concepts_summary}"
        f"{clarification_section}\n\n"
        f"## DesignIntent to Review\n```json\n{di_json}\n```"
    )

    critic_messages: list = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": critic_user_msg},
    ]

    critic_tool_log: list[tuple[str, dict]] = []

    for round_num in range(max_rounds):
        response = client.chat.completions.create(
            model=model,
            messages=critic_messages,
            tools=TOOLS if TOOLS else None,
        )
        choice = response.choices[0]
        critic_messages.append(choice.message)

        if choice.message.tool_calls:
            yield {"type": "phase", "phase": "critic",
                   "detail": f"Critic round {round_num + 1}: "
                             f"{len(choice.message.tool_calls)} tool call(s)"}
            for tc in choice.message.tool_calls:
                parsed_args, result = _execute_tool_raw(tc.function.name, tc.function.arguments)
                critic_tool_log.append((tc.function.name, parsed_args))

                yield {"type": "tool_call", "name": tc.function.name, "args": parsed_args}
                yield {"type": "tool_result", "name": tc.function.name, "result": result}

                critic_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            # Critic finished reasoning — show its text
            content = choice.message.content or ""
            if content.strip():
                yield {"type": "agent_text", "content": content}
            break

    # Now ask for the structured verdict
    critic_messages.append({
        "role": "user",
        "content": "Produce your structured CriticVerdict now."
    })

    verdict_response = client.beta.chat.completions.parse(
        model=model,
        messages=critic_messages,
        response_format=CriticVerdict,
    )
    verdict_msg = verdict_response.choices[0].message
    if verdict_msg.parsed:
        verdict = verdict_msg.parsed
    else:
        # Fallback: assume pass if the model refuses to produce a verdict
        verdict = CriticVerdict(passed=True, issues=[], summary="Critic could not produce a verdict; assuming pass.")

    yield {"type": "critic", "verdict": verdict.model_dump(), "attempt": 0}


def _extract_kg_queried_concepts(tool_log: list[tuple[str, dict]]) -> set[str]:
    """Extract which concepts were actually queried via KG tools."""
    queried = set()
    for tool_name, args in tool_log:
        if tool_name in _KG_TOOL_NAMES:
            # Grab whatever concept string was sent to the KG
            for key in ("query", "concept_name", "component_name", "function_name"):
                if key in args:
                    queried.add(args[key].lower().strip())
    return queried


# ---------------------------------------------------------------------------
# Phase 1 pipeline: explore + ask — yields events through Phase 1.75
# ---------------------------------------------------------------------------

def explore_and_ask(
    user_prompt: str,
    model: str = "o3-mini",
    max_tool_rounds: int = 15,
    max_grounding_rounds: int = 5,
    upstream_feedback: Optional[list[dict]] = None,
) -> Generator[AgentEvent, None, None]:
    """
    Run Phases 0 → 1 → 1.5 → 1.75 (disambiguation).

    Yields streaming events for the UI plus a final ``"clarification"`` event
    that contains the agent's questions and the full conversation history so
    the pipeline can be resumed via ``finalize_stream()``.

    Event types (in addition to the standard phase/tool/agent events):
        {"type": "clarification", "request": dict, "messages": list,
         "extracted": dict, "tool_log": list}
    """
    client = OpenAI()
    tool_log: list[tuple[str, dict]] = []

    messages: list = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Inject upstream feedback if this is a retry from downstream validation
    if upstream_feedback:
        items = "\n".join(
            f"  [{f.get('severity', '?').upper()}] {f.get('description', '')}\n"
            f"    Affects: {', '.join(f.get('affected_components', []))}\n"
            f"    Suggested: {f.get('suggested_action', '')}"
            for f in upstream_feedback
        )
        messages.append({
            "role": "user",
            "content": UPSTREAM_FEEDBACK_PROMPT.format(feedback_items=items),
        })
        yield {"type": "phase", "phase": "upstream_feedback",
               "detail": f"Injecting {len(upstream_feedback)} feedback item(s) from downstream"}

    # -------------------------------------------------------------------
    # Phase 0: LLM concept extraction from user prompt
    # -------------------------------------------------------------------
    yield {"type": "phase", "phase": "extraction",
           "detail": "Extracting photonic concepts from prompt..."}

    extracted = _extract_concepts_llm(client, user_prompt, model)

    yield {"type": "concepts",
           "components": extracted.components,
           "parameters": extracted.parameters,
           "specs": extracted.specs}

    # -------------------------------------------------------------------
    # Phase 1: Free agentic exploration
    # -------------------------------------------------------------------
    yield {"type": "phase", "phase": "exploration",
           "detail": f"Agent reasoning (max {max_tool_rounds} rounds)..."}

    for round_num in range(max_tool_rounds):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS if TOOLS else None,
        )
        choice = response.choices[0]
        messages.append(choice.message)

        if choice.message.tool_calls:
            yield {"type": "phase", "phase": "exploration",
                   "detail": f"Round {round_num + 1}: {len(choice.message.tool_calls)} tool call(s)"}
            for tc in choice.message.tool_calls:
                parsed_args, result = _execute_tool_raw(tc.function.name, tc.function.arguments)
                tool_log.append((tc.function.name, parsed_args))

                yield {"type": "tool_call", "name": tc.function.name, "args": parsed_args}
                yield {"type": "tool_result", "name": tc.function.name, "result": result}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            content = choice.message.content or ""
            yield {"type": "agent_text", "content": content}
            break

    # -------------------------------------------------------------------
    # Phase 1.5: KG Grounding Gate
    # -------------------------------------------------------------------
    if _KG_AVAILABLE:
        kg_calls = [t for t in tool_log if t[0] in _KG_TOOL_NAMES]
        mentioned_filtered = set(extracted.components)

        if len(kg_calls) == 0 and len(mentioned_filtered) > 0:
            ungrounded_list = sorted(mentioned_filtered)
            yield {"type": "grounding",
                   "detail": f"Agent never consulted KG. Self-audit for {len(ungrounded_list)} concept(s).",
                   "ungrounded": ungrounded_list}

            grounding_msg = GROUNDING_GATE_PROMPT.format(
                ungrounded_items="\n".join(f"  - {c}" for c in ungrounded_list)
            )
            messages.append({"role": "user", "content": grounding_msg})

            yield {"type": "phase", "phase": "grounding",
                   "detail": f"KG grounding gate triggered (max {max_grounding_rounds} rounds)..."}

            for ground_round in range(max_grounding_rounds):
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=TOOLS if TOOLS else None,
                )
                choice = response.choices[0]
                messages.append(choice.message)

                if choice.message.tool_calls:
                    yield {"type": "phase", "phase": "grounding",
                           "detail": f"Grounding round {ground_round + 1}: "
                                     f"{len(choice.message.tool_calls)} tool call(s)"}
                    for tc in choice.message.tool_calls:
                        parsed_args, result = _execute_tool_raw(tc.function.name, tc.function.arguments)
                        tool_log.append((tc.function.name, parsed_args))
                        yield {"type": "tool_call", "name": tc.function.name, "args": parsed_args}
                        yield {"type": "tool_result", "name": tc.function.name, "result": result}
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })
                else:
                    content = choice.message.content or ""
                    yield {"type": "agent_text", "content": content}
                    break
        else:
            if len(kg_calls) > 0:
                kg_queried = _extract_kg_queried_concepts(tool_log)
                yield {"type": "phase", "phase": "grounding",
                       "detail": f"Agent consulted KG for: {sorted(kg_queried)}. Gate not triggered."}
            else:
                yield {"type": "phase", "phase": "grounding",
                       "detail": "No non-trivial concepts found. Gate not triggered."}
    else:
        yield {"type": "phase", "phase": "grounding",
               "detail": "KG unavailable, skipping."}

    # -------------------------------------------------------------------
    # Tool stats
    # -------------------------------------------------------------------
    total_kg = len([t for t in tool_log if t[0] in _KG_TOOL_NAMES])
    total_pdk = len([t for t in tool_log if t[0] not in _KG_TOOL_NAMES])
    yield {"type": "stats", "total": len(tool_log), "pdk": total_pdk, "kg": total_kg}

    # -------------------------------------------------------------------
    # Phase 1.75: Disambiguation — agent generates clarification questions
    # -------------------------------------------------------------------
    yield {"type": "phase", "phase": "disambiguation",
           "detail": "Agent assessing remaining uncertainties..."}

    messages.append({"role": "user", "content": DISAMBIGUATION_PROMPT})

    disambig_response = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=ClarificationRequest,
    )
    disambig_msg = disambig_response.choices[0].message
    if disambig_msg.parsed:
        clarification_req = disambig_msg.parsed
    else:
        clarification_req = ClarificationRequest(questions=[], ready_to_proceed=True)

    # Serialise messages for session state storage (ChatCompletionMessage → dict)
    serializable_messages = _serialise_messages(messages)

    yield {
        "type": "clarification",
        "request": clarification_req.model_dump(),
        "messages": serializable_messages,
        "extracted": extracted.model_dump(),
        "tool_log": tool_log,
    }


def _serialise_messages(messages: list) -> list[dict]:
    """Convert a message list to JSON-safe dicts (handles OpenAI objects)."""
    out = []
    for msg in messages:
        if isinstance(msg, dict):
            out.append(msg)
        else:
            d: dict = {"role": msg.role, "content": msg.content or ""}
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            if hasattr(msg, "refusal") and msg.refusal:
                d["refusal"] = msg.refusal
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# Phase 2+ pipeline: finalize — takes saved state and produces DesignIntent
# ---------------------------------------------------------------------------

def finalize_stream(
    messages: list,
    tool_log: list[tuple[str, dict]],
    user_prompt: str,
    extracted_dict: dict,
    model: str = "o3-mini",
    max_tool_rounds: int = 10,
    max_critic_rounds: int = 2,
    clarifications: Optional[dict[str, str]] = None,
    upstream_feedback: Optional[str] = None,
) -> Generator[AgentEvent, None, None]:
    """
    Run Phases 2 → 3 (with optional clarification injection + re-exploration).

    Parameters
    ----------
    messages : list
        Conversation history from ``explore_and_ask()``.
    tool_log : list
        Tool call log from ``explore_and_ask()``.
    user_prompt : str
        Original user prompt (needed for critic context).
    extracted_dict : dict
        Serialised ``ExtractedConcepts`` from Phase 0.
    clarifications : dict, optional
        ``{ambiguity_question: user_answer}`` pairs from the disambiguation form.
        If provided, injected before Phase 2.
    upstream_feedback : str, optional
        Free-text feedback from downstream (e.g. schematic review gate).
        If provided, injected as a user message before clarifications.

    Event types (same as explore_and_ask plus):
        {"type": "done", "result": DesignIntent, "messages": list}
        {"type": "error", "message": str}
    """
    client = OpenAI()
    extracted = ExtractedConcepts(**extracted_dict)

    # -------------------------------------------------------------------
    # Inject upstream/schematic feedback (if any)
    # -------------------------------------------------------------------
    if upstream_feedback:
        messages.append({"role": "user", "content": upstream_feedback})
        yield {"type": "phase", "phase": "upstream_feedback",
               "detail": "Injecting user schematic feedback for revision"}

    # -------------------------------------------------------------------
    # Inject user clarifications (if any) + brief re-exploration
    # -------------------------------------------------------------------
    if clarifications:
        items = "\n".join(
            f"  - **Q:** {q}\n    **A:** {a}" for q, a in clarifications.items()
        )
        injection = CLARIFICATION_INJECTION_PROMPT.format(clarification_items=items)
        messages.append({"role": "user", "content": injection})

        yield {"type": "phase", "phase": "clarification_update",
               "detail": f"Injecting {len(clarifications)} clarification(s) and re-exploring..."}

        for round_num in range(max_tool_rounds):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS if TOOLS else None,
            )
            choice = response.choices[0]
            messages.append(choice.message)

            if choice.message.tool_calls:
                yield {"type": "phase", "phase": "clarification_update",
                       "detail": f"Round {round_num + 1}: {len(choice.message.tool_calls)} tool call(s)"}
                for tc in choice.message.tool_calls:
                    parsed_args, result = _execute_tool_raw(tc.function.name, tc.function.arguments)
                    tool_log.append((tc.function.name, parsed_args))

                    yield {"type": "tool_call", "name": tc.function.name, "args": parsed_args}
                    yield {"type": "tool_result", "name": tc.function.name, "result": result}

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
            else:
                content = choice.message.content or ""
                if content.strip():
                    yield {"type": "agent_text", "content": content}
                break

    # -------------------------------------------------------------------
    # Outer loop: Phase 2 → 3 (critic), repeat on failure
    # -------------------------------------------------------------------
    total_attempts = 1 + max_critic_rounds
    design_intent: Optional[DesignIntent] = None

    for attempt in range(total_attempts):
        attempt_label = f" (attempt {attempt + 1}/{total_attempts})" if total_attempts > 1 else ""

        # ---------------------------------------------------------------
        # Phase 2: Structured output extraction
        # ---------------------------------------------------------------
        yield {"type": "phase", "phase": "structuring",
               "detail": f"Producing structured DesignIntent{attempt_label}..."}

        messages.append({"role": "user", "content": STRUCTURING_PROMPT})

        structured_response = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=DesignIntent,
        )
        struct_msg = structured_response.choices[0].message
        if not struct_msg.parsed:
            yield {"type": "error", "message": f"LLM refused to produce DesignIntent: {struct_msg.refusal}"}
            return

        design_intent = struct_msg.parsed

        # ---------------------------------------------------------------
        # Phase 3: Critic review
        # ---------------------------------------------------------------
        if attempt == total_attempts - 1:
            break

        yield {"type": "phase", "phase": "critic",
               "detail": f"Critic reviewing DesignIntent{attempt_label}..."}

        critic_verdict: Optional[CriticVerdict] = None
        for critic_event in _run_critic(
            client, user_prompt, extracted, design_intent, model,
            clarifications=clarifications,
        ):
            if critic_event["type"] == "critic":
                critic_event["attempt"] = attempt + 1
                critic_verdict_data = critic_event["verdict"]
                critic_verdict = CriticVerdict(**critic_verdict_data)
            yield critic_event

        if critic_verdict is None:
            break

        if critic_verdict.passed:
            yield {"type": "phase", "phase": "critic",
                   "detail": f"Critic PASSED: {critic_verdict.summary}"}
            break

        issues_text = "\n".join(
            f"  [{issue.severity.upper()}] {issue.description} "
            f"(affects: {', '.join(issue.affected_components) or 'global'})"
            for issue in critic_verdict.issues
        )
        feedback = CRITIC_FEEDBACK_PROMPT.format(
            attempt=attempt + 1,
            max_attempts=total_attempts,
            issues_text=issues_text,
            summary=critic_verdict.summary,
        )
        messages.append({"role": "user", "content": feedback})

        yield {"type": "phase", "phase": "critic",
               "detail": f"Critic FAILED (attempt {attempt + 1}/{total_attempts}). "
                         f"{len(critic_verdict.issues)} issue(s). Retrying..."}

    # -------------------------------------------------------------------
    # Yield final result
    # -------------------------------------------------------------------
    if design_intent is not None:
        serializable_messages = _serialise_messages(messages)
        yield {"type": "done", "result": design_intent, "messages": serializable_messages}
    else:
        yield {"type": "error", "message": "Agent loop ended without producing a DesignIntent"}


# ---------------------------------------------------------------------------
# Backward-compatible wrappers
# ---------------------------------------------------------------------------

def interpret_stream(
    user_prompt: str,
    model: str = "o3-mini",
    max_tool_rounds: int = 15,
    max_grounding_rounds: int = 5,
    max_critic_rounds: int = 2,
) -> Generator[AgentEvent, None, None]:
    """
    Full streaming interpreter — runs all phases end-to-end (no user disambiguation pause).

    Kept for backward compatibility and non-interactive use. Equivalent to calling
    ``explore_and_ask()`` followed by ``finalize_stream()`` with no clarifications.

    Event types:
        {"type": "phase",       "phase": str, "detail": str}
        {"type": "concepts",    "components": list[str], "parameters": list[str], "specs": list[str]}
        {"type": "tool_call",   "name": str,  "args": dict}
        {"type": "tool_result", "name": str,  "result": str}
        {"type": "agent_text",  "content": str}
        {"type": "grounding",   "detail": str, "ungrounded": list[str]}
        {"type": "stats",       "total": int, "pdk": int, "kg": int}
        {"type": "clarification", ...}
        {"type": "critic",      "verdict": dict, "attempt": int}
        {"type": "done",        "result": DesignIntent}
        {"type": "error",       "message": str}
    """
    saved_state: Optional[dict] = None

    for event in explore_and_ask(
        user_prompt, model, max_tool_rounds, max_grounding_rounds,
    ):
        if event["type"] == "clarification":
            saved_state = event
        else:
            yield event

    if saved_state is None:
        yield {"type": "error", "message": "explore_and_ask ended without producing clarification state"}
        return

    # Proceed directly to finalization — no user answers
    for event in finalize_stream(
        messages=saved_state["messages"],
        tool_log=saved_state["tool_log"],
        user_prompt=user_prompt,
        extracted_dict=saved_state["extracted"],
        model=model,
        max_tool_rounds=max_tool_rounds,
        max_critic_rounds=max_critic_rounds,
        clarifications=None,
    ):
        yield event


def interpret(
    user_prompt: str,
    model: str = "o3-mini",
    max_tool_rounds: int = 15,
    max_grounding_rounds: int = 5,
    max_critic_rounds: int = 2,
    verbose: bool = True,
) -> DesignIntent:
    """
    Interpret a user prompt into a structured DesignIntent (blocking).

    Internally calls interpret_stream() and consumes all events.
    """
    result: Optional[DesignIntent] = None

    for event in interpret_stream(
        user_prompt, model, max_tool_rounds, max_grounding_rounds, max_critic_rounds
    ):
        etype = event["type"]

        if verbose:
            if etype == "phase":
                phase_label = {"extraction": "Phase 0", "exploration": "Phase 1",
                               "grounding": "Phase 1.5", "disambiguation": "Phase 1.75",
                               "clarification_update": "Phase 1.75+",
                               "structuring": "Phase 2",
                               "critic": "Phase 3"}.get(event["phase"], event["phase"])
                print(f"{phase_label}: {event['detail']}")
            elif etype == "concepts":
                print(f"  Components: {event['components']}")
                print(f"  Parameters: {event['parameters']}")
                print(f"  Specs: {event['specs']}")
            elif etype == "tool_call":
                print(f"    🔧 {event['name']}({json.dumps(event['args'], ensure_ascii=False)})")
            elif etype == "tool_result":
                preview = event["result"][:200] + "..." if len(event["result"]) > 200 else event["result"]
                print(f"    → {preview}")
            elif etype == "agent_text":
                print(f"  Agent: {event['content'][:150]}...")
            elif etype == "grounding":
                print(f"  Grounding: {event['detail']}")
            elif etype == "stats":
                print(f"  Total tool calls: {event['total']} ({event['pdk']} PDK, {event['kg']} KG)")
            elif etype == "clarification":
                req = event["request"]
                n_q = len(req.get("questions", []))
                rtp = req.get("ready_to_proceed", True)
                print(f"  Disambiguation: {n_q} question(s), ready_to_proceed={rtp}")
            elif etype == "critic":
                v = event["verdict"]
                status = "PASSED" if v["passed"] else "FAILED"
                n_issues = len(v.get("issues", []))
                print(f"  Critic {status} (attempt {event['attempt']}): "
                      f"{v['summary'][:120]} [{n_issues} issue(s)]")
            elif etype == "done":
                di = event["result"]
                print(f"  → {len(di.components)} components, "
                      f"{len(di.connections)} connections, "
                      f"{len(di.ambiguities)} ambiguities")
            elif etype == "error":
                print(f"  ERROR: {event['message']}")

        if etype == "done":
            result = event["result"]
        elif etype == "error":
            raise ValueError(event["message"])

    if result is None:
        raise ValueError("interpret_stream ended without producing a DesignIntent")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    prompt = " ".join(sys.argv[1:]) or "A 1x2 beamsplitter connected to two MZI modulators with heaters"
    result = interpret(prompt, verbose=True)

    output = {
        "summary": result.summary(),
        "full": result.full(),
        "legacy_pretemplate": result.to_pretemplate(),
    }
    out_path = "intent_output.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Result written to {out_path}")
