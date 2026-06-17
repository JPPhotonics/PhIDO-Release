"""
Clingo-based topology validator for the PhIDO pipeline (Checkpoint A).

Converts a DesignIntent into Clingo facts, runs the solver against
architecture_rules.lp, and translates error atoms into PipelineFeedback.

Graceful degradation: if clingo is not installed, validate_topology()
returns [] with a logged warning so the pipeline continues.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mcp_servers.models import DesignIntent, PipelineFeedback

logger = logging.getLogger(__name__)

try:
    import clingo as _clingo

    _CLINGO_AVAILABLE = True
except ImportError:
    _clingo = None  # type: ignore[assignment]
    _CLINGO_AVAILABLE = False

RULES_PATH = Path(__file__).parent.parent / "docs" / "policies" / "architecture_rules.lp"

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ClingoResult:
    satisfiable: bool
    errors: list[dict] = field(default_factory=list)
    answer_set: list[str] = field(default_factory=list)
    solve_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Error message templates  (keyed by error-atom functor)
# ---------------------------------------------------------------------------

_ERROR_MESSAGES: dict[str, str] = {
    "mzi_wrong_splitter_count":
        "MZI requires exactly 1 splitter, found {0}.",
    "mzi_wrong_combiner_count":
        "MZI requires exactly 1 combiner, found {0}.",
    "mzi_disconnected":
        "MZI splitter is not connected to combiner (broken interferometric path).",
    "tree_not_power_of_2":
        "Splitter tree output count N={0} is not a power of 2.",
    "tree_wrong_splitters":
        "Splitter tree with N={0} has {1} splitters (expected {2}).",
    "benes_wrong_switch_count":
        "Benes network N={0} has {1} switches (expected {2}).",
    "clements_wrong_mzi_count":
        "Clements mesh N={0} has {1} MZI blocks (expected {2}).",
    "reck_wrong_mzi_count":
        "Reck mesh N={0} has {1} MZI blocks (expected {2}).",
    "wdm_insufficient_rings":
        "WDM with N={0} channels has only {1} ring resonators (need >= {0}).",
    "qpsk_wrong_splitter_count":
        "QPSK modulator requires exactly 1 splitter, found {0}.",
    "qpsk_wrong_combiner_count":
        "QPSK modulator requires exactly 1 combiner, found {0}.",
    "qpsk_wrong_mzm_count":
        "QPSK modulator requires exactly 2 MZMs, found {0}.",
    "qpsk_missing_phase_shifter":
        "QPSK modulator is missing a phase shifter on the Q path.",
    "unsatisfied_requirement":
        "Requirement {0} has no design element satisfying it.",
    "unaddressed_requirement":
        "Requirement {0} is explicitly marked as unaddressed.",
}

_TREE_EXPECTED = {2: 1, 4: 3, 8: 7, 16: 15, 32: 31, 64: 63}
_BENES_EXPECTED = {4: 6, 8: 20}
_CLEMENTS_EXPECTED = {4: 6, 8: 28}


# ---------------------------------------------------------------------------
# Fact generation
# ---------------------------------------------------------------------------

def _safe_id(s: str) -> str:
    """Lowercase an identifier and wrap in quotes for Clingo."""
    return f'"{s.lower()}"'


def _safe_atom(s: str) -> str:
    """Convert a string to a safe Clingo atom (lowercase, alnum + underscore)."""
    return re.sub(r"[^a-z0-9_]", "_", s.lower())


def _resolve_component_type(comp) -> str:
    """Determine the Clingo component type from the best available field.

    Clingo rules count functional roles where they disambiguate generic devices
    (for example a coupler used as a splitter), but an explicit canonical
    ``component_type`` should win over broad roles such as ``modulator``.
    """
    explicit_type = _safe_atom(comp.component_type) if comp.component_type else None
    if explicit_type and explicit_type not in {"coupler", "mmi"}:
        return explicit_type

    if comp.role:
        role_lower = comp.role.lower()
        for keyword, ctype in _ROLE_TO_TYPE.items():
            if keyword in role_lower:
                return ctype

    if explicit_type:
        return explicit_type

    return _infer_component_type(comp)


def _facts_from_structured_fields(di: DesignIntent) -> list[str]:
    """Emit component/3 and connection/2 facts from enriched fields."""
    facts: list[str] = []
    for comp in di.components:
        ctype = _resolve_component_type(comp)
        stype = f'"{_safe_atom(comp.sub_type)}"' if comp.sub_type else '"none"'
        if not stype or stype == '"none"':
            stype = f'"{_infer_sub_type(comp)}"'
        facts.append(f"component({_safe_id(comp.id)}, {ctype}, {stype}).")
    for conn in di.connections:
        facts.append(f"connection({_safe_id(conn.from_component)}, {_safe_id(conn.to_component)}).")
    return facts


def _facts_from_architecture_metadata(di: DesignIntent) -> list[str]:
    """Emit architecture/1 and n_value/1 facts."""
    facts: list[str] = []
    if di.architecture_type:
        facts.append(f"architecture({_safe_atom(di.architecture_type)}).")
    if di.n_value is not None:
        facts.append(f"n_value({di.n_value}).")
    return facts


def _facts_from_requirement_traces(di: DesignIntent) -> list[str]:
    """Emit requirement/1 and requirement_trace/2 facts."""
    facts: list[str] = []
    if di.requirement_manifest:
        for req in di.requirement_manifest.requirements:
            facts.append(f"requirement({_safe_id(req.id)}).")
    for trace in di.requirement_traces:
        sat = _safe_atom(trace.satisfaction_type)
        facts.append(f"requirement_trace({_safe_id(trace.requirement_id)}, {sat}).")
    return facts


# -- Fallback path for legacy DesignIntent without enriched fields ----------

_ARCH_KEYWORD_TO_ATOM = {
    "mach-zehnder interferometer": "mzi",
    "mzi": "mzi",
    "benes": "benes",
    "clements": "clements",
    "reck": "reck",
    "qpsk": "qpsk",
    "wdm demux": "wdm_demux",
    "wdm demultiplexer": "wdm_demux",
    "wdm multiplexer": "wdm_mux",
    "wdm mux": "wdm_mux",
    "splitter tree": "splitter_tree",
    "power splitter": "splitter_tree",
    "crossbar": "crossbar",
    "spanke": "spanke",
    "ring filter": "ring_filter",
}

_DESC_TO_COMPONENT_TYPE = {
    "splitter": "splitter",
    "mmi": "splitter",
    "combiner": "combiner",
    "mach-zehnder modulator": "mzm",
    "mzm": "mzm",
    "phase shifter": "phase_shifter",
    "ring resonator": "ring_resonator",
    "ring": "ring_resonator",
    "waveguide": "waveguide",
    "crossing": "crossing",
    "detector": "detector",
    "grating coupler": "grating_coupler",
    "coupler": "coupler",
}

_ROLE_TO_TYPE = {
    "splitter": "splitter",
    "combiner": "combiner",
    "modulator": "mzm",
    "phase_shifter": "phase_shifter",
    "filter": "ring_resonator",
    "detector": "detector",
    "coupler": "coupler",
    "waveguide": "waveguide",
}


def _infer_component_type(comp) -> str:
    """Best-effort component type from description and role."""
    desc = comp.description.lower()
    for keyword, ctype in _DESC_TO_COMPONENT_TYPE.items():
        if keyword in desc:
            return ctype
    if comp.role:
        role_lower = comp.role.lower()
        for keyword, ctype in _ROLE_TO_TYPE.items():
            if keyword in role_lower:
                return ctype
    return "unknown"


def _infer_sub_type(comp) -> str:
    """Best-effort sub_type from description."""
    desc = comp.description.lower()
    if "90" in desc and "phase" in desc:
        return "90_degree"
    if "mmi" in desc:
        return "mmi"
    if "directional coupler" in desc:
        return "directional_coupler"
    if "add-drop" in desc or "add/drop" in desc:
        return "add_drop"
    if "all-pass" in desc or "all pass" in desc:
        return "all_pass"
    return "none"


def _extract_n_value_from_text(di: DesignIntent) -> Optional[int]:
    """Extract N from title/summary using regex patterns."""
    combined = f"{di.title} {di.brief_summary}".lower()
    for pattern in [
        r"(\d+)\s*x\s*(\d+)",
        r"1\s*[x×]\s*(\d+)",
        r"(\d+)\s+outputs?",
        r"(\d+)\s*-?\s*channel",
    ]:
        m = re.search(pattern, combined)
        if m:
            val = int(m.groups()[-1])
            if val >= 2:
                return val
    return None


def _fallback_architecture_detection(di: DesignIntent) -> list[str]:
    """Keyword-based fact generation for legacy DesignIntent objects."""
    facts: list[str] = []
    text = (
        f"{di.title} {di.brief_summary} "
        + " ".join(c.description for c in di.components)
    ).lower()

    detected_atoms: set[str] = set()
    for kw, atom in _ARCH_KEYWORD_TO_ATOM.items():
        if kw in text:
            detected_atoms.add(atom)

    for atom in detected_atoms:
        facts.append(f"architecture({atom}).")

    n = _extract_n_value_from_text(di)
    if n is not None:
        facts.append(f"n_value({n}).")

    for comp in di.components:
        ctype = _infer_component_type(comp)
        stype = f'"{_infer_sub_type(comp)}"'
        facts.append(f'component({_safe_id(comp.id)}, {ctype}, {stype}).')

    for conn in di.connections:
        facts.append(f"connection({_safe_id(conn.from_component)}, {_safe_id(conn.to_component)}).")

    return facts


def design_intent_to_facts(di: DesignIntent) -> str:
    """Convert a DesignIntent to a Clingo fact string.

    Uses enriched fields when available, falls back to keyword detection.
    """
    facts: list[str] = []

    if di.architecture_type:
        facts.extend(_facts_from_architecture_metadata(di))
        facts.extend(_facts_from_structured_fields(di))
    else:
        facts.extend(_fallback_architecture_detection(di))

    facts.extend(_facts_from_requirement_traces(di))
    return "\n".join(facts)


# ---------------------------------------------------------------------------
# Solver invocation
# ---------------------------------------------------------------------------

def run_clingo(facts: str, rules_path: Optional[str] = None) -> ClingoResult:
    """Run the Clingo solver and return parsed results."""
    if not _CLINGO_AVAILABLE:
        logger.warning("Clingo not installed — returning empty result.")
        return ClingoResult(satisfiable=True)

    rpath = rules_path or str(RULES_PATH)

    ctl = _clingo.Control(["0"])  # enumerate all models
    ctl.load(rpath)
    ctl.add("base", [], facts)
    ctl.ground([("base", [])])

    answer_set: list[str] = []
    satisfiable = False

    t0 = time.perf_counter()
    with ctl.solve(yield_=True) as handle:  # type: ignore[union-attr]
        for model in handle:
            answer_set = [str(a) for a in model.symbols(shown=True)]
        sat_result = handle.get()
        satisfiable = sat_result.satisfiable  # type: ignore[union-attr]
    solve_ms = (time.perf_counter() - t0) * 1000

    errors = _parse_errors(answer_set)
    return ClingoResult(
        satisfiable=satisfiable,
        errors=errors,
        answer_set=answer_set,
        solve_time_ms=solve_ms,
    )


def _parse_errors(answer_set: list[str]) -> list[dict]:
    """Extract error(...) atoms from the answer set."""
    errors: list[dict] = []
    for atom_str in answer_set:
        if not atom_str.startswith("error("):
            continue
        inner = atom_str[len("error("):-1]
        args = [a.strip().strip('"') for a in inner.split(",")]
        code = args[0]
        detail_args = args[1:]

        template = _ERROR_MESSAGES.get(code)
        if template:
            # For tree_wrong_splitters, add expected count from lookup
            if code == "tree_wrong_splitters" and len(detail_args) >= 1:
                try:
                    n = int(detail_args[0])
                    expected = _TREE_EXPECTED.get(n, "?")
                    detail_args.append(str(expected))
                except ValueError:
                    pass
            elif code == "benes_wrong_switch_count" and len(detail_args) >= 1:
                try:
                    n = int(detail_args[0])
                    expected = _BENES_EXPECTED.get(n, "?")
                    detail_args.append(str(expected))
                except ValueError:
                    pass
            elif code in ("clements_wrong_mzi_count", "reck_wrong_mzi_count") and len(detail_args) >= 1:
                try:
                    n = int(detail_args[0])
                    expected = _CLEMENTS_EXPECTED.get(n, "?")
                    detail_args.append(str(expected))
                except ValueError:
                    pass
            try:
                message = template.format(*detail_args)
            except (IndexError, KeyError):
                message = f"{code}: {', '.join(detail_args)}"
        else:
            message = f"{code}: {', '.join(detail_args)}" if detail_args else code

        errors.append({"code": code, "args": detail_args, "message": message})
    return errors


# ---------------------------------------------------------------------------
# Top-level validation function
# ---------------------------------------------------------------------------

def validate_topology(di: DesignIntent) -> list[PipelineFeedback]:
    """Checkpoint A: Validate DesignIntent topology using Clingo.

    Returns PipelineFeedback items for each topology error found.
    Returns [] if Clingo is not installed or no errors are detected.
    """
    if not _CLINGO_AVAILABLE:
        logger.warning("Clingo not installed, skipping topology validation.")
        return []

    # Diagnostic: show what the interpreter provided vs what Clingo receives
    print("── Clingo Validator Diagnostics ──")
    print(f"  architecture_type = {di.architecture_type!r}")
    print(f"  n_value           = {di.n_value!r}")
    for comp in di.components:
        resolved = _resolve_component_type(comp)
        print(f"  {comp.id}: role={comp.role!r}  component_type={comp.component_type!r}  → clingo_type={resolved!r}")

    facts = design_intent_to_facts(di)
    print(f"── Clingo facts ──\n{facts}\n── end facts ──")
    logger.debug("Clingo facts:\n%s", facts)

    result = run_clingo(facts)

    if not result.errors:
        logger.info("Topology validation passed (%.1fms).", result.solve_time_ms)
        return []

    logger.info(
        "Topology validation found %d error(s) (%.1fms).",
        len(result.errors), result.solve_time_ms,
    )

    feedback: list[PipelineFeedback] = []
    for err in result.errors:
        feedback.append(PipelineFeedback(
            source_phase="topology_gate",
            target_phase="interpreter",
            severity="fundamental",
            description=err["message"],
            affected_components=["global"],
            suggested_action="Revise the design to fix the topology error.",
            context={
                "clingo_error_code": err["code"],
                "clingo_error_args": err["args"],
                "solve_time_ms": result.solve_time_ms,
            },
        ))
    return feedback
