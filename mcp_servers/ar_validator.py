"""
Automated Reasoning Validator — Bedrock AR integration for PhIDO pipeline.

Provides two validation checkpoints:
  Checkpoint A (architecture gate): Validates DesignIntent architecture against
      Tier 6 policy rules (MZI structure, binary tree scaling, switching networks,
      WDM, QPSK modulator).
  Checkpoint B (parameter gate): Validates resolved circuit DSL parameters against
      Tier 6 physical bound rules (widths, radii, gaps, lengths, splitting ratios).

Both checkpoints call the AWS Bedrock ApplyGuardrail API and return PipelineFeedback
objects. On failure (missing credentials, disabled, API error), they degrade
gracefully and return an empty list so the pipeline continues.
"""

import json
import logging
import os
import re
from typing import Optional

from mcp_servers.models import DesignIntent, PipelineFeedback

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity ordering for AR finding types (lower = worse)
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = {
    "tooComplex": 0,
    "translationAmbiguous": 0,
    "impossible": 1,
    "invalid": 2,
    "satisfiable": 3,
    "valid": 4,
    "noTranslations": 5,
}

# ---------------------------------------------------------------------------
# boto3 client (lazy singleton)
# ---------------------------------------------------------------------------

_bedrock_client = None


def _get_ar_client():
    """Return a lazily-initialized bedrock-runtime client."""
    global _bedrock_client
    if _bedrock_client is not None:
        return _bedrock_client
    try:
        import boto3
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        _bedrock_client = boto3.client("bedrock-runtime", region_name=region)
        return _bedrock_client
    except Exception as exc:
        logger.warning("Could not create Bedrock client: %s", exc)
        return None


def _ar_enabled() -> bool:
    """Check whether AR validation is enabled and configured."""
    if os.environ.get("AR_ENABLED", "true").lower() != "true":
        return False
    if not os.environ.get("AR_GUARDRAIL_ID"):
        return False
    if not os.environ.get("AR_GUARDRAIL_VERSION"):
        return False
    return True


# ---------------------------------------------------------------------------
# Low-level API call + finding parser
# ---------------------------------------------------------------------------

def _call_apply_guardrail(
    input_text: str,
    output_text: str,
) -> Optional[dict]:
    """Call ApplyGuardrail and return the raw response, or None on error."""
    client = _get_ar_client()
    if client is None:
        return None

    guardrail_id = os.environ["AR_GUARDRAIL_ID"]
    guardrail_version = os.environ["AR_GUARDRAIL_VERSION"]

    combined = f"User: {input_text}\nAssistant: {output_text}"
    try:
        response = client.apply_guardrail(
            guardrailIdentifier=guardrail_id,
            guardrailVersion=guardrail_version,
            source="OUTPUT",
            content=[{"text": {"text": combined}}],
        )
        return response
    except Exception as exc:
        logger.warning("ApplyGuardrail call failed: %s", exc)
        return None


def _get_finding_type(finding: dict) -> tuple[Optional[str], Optional[dict]]:
    """Extract the finding type and its data from an AR finding union."""
    for ftype in (
        "valid", "invalid", "satisfiable", "impossible",
        "translationAmbiguous", "tooComplex", "noTranslations",
    ):
        if ftype in finding:
            return ftype, finding[ftype]
    return None, None


def _parse_findings(response: dict) -> list[dict]:
    """Parse AR findings from an ApplyGuardrail response.

    Returns a list of dicts with keys: type, data, premises, claims,
    contradicting_rules, supporting_rules.
    """
    parsed = []
    for assessment in response.get("assessments", []):
        ar = assessment.get("automatedReasoningPolicy", {})
        for finding in ar.get("findings", []):
            ftype, fdata = _get_finding_type(finding)
            if ftype is None:
                continue

            translation = (fdata or {}).get("translation", {})
            entry = {
                "type": ftype,
                "data": fdata,
                "premises": translation.get("premises", []),
                "claims": translation.get("claims", []),
                "untranslated_premises": translation.get(
                    "untranslatedPremises", []
                ),
                "untranslated_claims": translation.get(
                    "untranslatedClaims", []
                ),
                "contradicting_rules": (fdata or {}).get(
                    "contradictingRules", []
                ),
                "supporting_rules": (fdata or {}).get("supportingRules", []),
            }
            parsed.append(entry)
    return parsed


def _get_aggregate_result(findings: list[dict]) -> Optional[str]:
    """Return the worst finding type from a list of parsed findings."""
    worst = None
    worst_severity = float("inf")
    for f in findings:
        sev = _SEVERITY_ORDER.get(f["type"], 0)
        if sev < worst_severity:
            worst_severity = sev
            worst = f["type"]
    return worst


# ---------------------------------------------------------------------------
# Checkpoint A: Architecture serializer
# ---------------------------------------------------------------------------

_ARCH_KEYWORDS = {
    "mzi": "Mach-Zehnder interferometer",
    "mach-zehnder": "Mach-Zehnder interferometer",
    "benes": "Benes switching network",
    "clements": "Clements unitary mesh",
    "reck": "Reck unitary mesh",
    "qpsk": "QPSK modulator",
    "wdm": "WDM",
    "demux": "WDM demultiplexer",
    "demultiplexer": "WDM demultiplexer",
    "multiplexer": "WDM multiplexer",
    "splitter tree": "power splitter tree",
    "power splitter": "power splitter tree",
    "crossbar": "crossbar switching network",
    "spanke": "Spanke switching network",
    "ring resonator": "ring resonator",
}


def _detect_architecture(design_intent: DesignIntent) -> list[str]:
    """Identify architecture keywords from the DesignIntent."""
    text = (
        f"{design_intent.title} {design_intent.brief_summary} "
        + " ".join(c.description for c in design_intent.components)
    ).lower()
    detected = []
    for kw, label in _ARCH_KEYWORDS.items():
        if kw in text and label not in detected:
            detected.append(label)
    return detected


def _extract_n_value(design_intent: DesignIntent) -> Optional[int]:
    """Try to extract the N value (output count / port size) from the design."""
    title_lower = design_intent.title.lower()
    summary_lower = design_intent.brief_summary.lower()
    combined = f"{title_lower} {summary_lower}"

    for pattern in [
        r"(\d+)\s*x\s*(\d+)",
        r"1\s*[x×]\s*(\d+)",
        r"(\d+)\s+outputs?",
        r"(\d+)\s*-?\s*channel",
    ]:
        m = re.search(pattern, combined)
        if m:
            groups = m.groups()
            val = int(groups[-1])
            if val >= 2:
                return val
    return None


def serialize_architecture_claims(
    design_intent: DesignIntent,
) -> tuple[str, str]:
    """Convert a DesignIntent into (Input, Output) text for AR architecture validation.

    The Input describes the architectural facts as premises.
    The Output states validity claims for AR to check.
    """
    architectures = _detect_architecture(design_intent)
    n_value = _extract_n_value(design_intent)
    num_components = len(design_intent.components)
    num_connections = len(design_intent.connections)

    premises = []
    claims = []

    premises.append(
        f"The circuit is titled \"{design_intent.title}\". "
        f"It has {num_components} components and {num_connections} connections."
    )
    premises.append(f"Brief description: {design_intent.brief_summary}")

    # Summarise components by (description, role, port_config).
    # The architecture-specific enrichment block below provides the
    # AR-critical variable bindings; the component listing gives context.
    from collections import defaultdict
    comp_groups: dict[tuple, list] = defaultdict(list)
    for comp in design_intent.components:
        key = (comp.description, comp.role or "", comp.port_config or "")
        comp_groups[key].append(comp)

    for (desc, role, pcfg), members in comp_groups.items():
        count = len(members)
        if count > 1:
            parts = [f"{count}x \"{desc}\""]
        else:
            parts = [f"\"{desc}\""]
        if role:
            parts.append(f"role={role}")
        if pcfg:
            parts.append(pcfg)
        seen_specs: set[tuple[str, str]] = set()
        for m in members:
            for spec in m.specs:
                if (spec.key, spec.value) not in seen_specs:
                    seen_specs.add((spec.key, spec.value))
                    parts.append(f"{spec.key}={spec.value}")
        premises.append("Component: " + ", ".join(parts) + ".")

    if num_connections <= 4:
        for conn in design_intent.connections:
            premises.append(
                f"{conn.from_component}->{conn.to_component}: "
                f"{conn.description}"
            )
    else:
        premises.append(f"The circuit has {num_connections} internal connections.")

    # Architecture-specific premise enrichment
    roles = [c.role.lower() for c in design_intent.components if c.role]
    descriptions = [c.description.lower() for c in design_intent.components]
    all_desc = " ".join(descriptions)

    splitter_count = sum(
        1 for r in roles if "splitter" in r
    ) + sum(1 for d in descriptions if "splitter" in d or "mmi" in d)
    combiner_count = sum(1 for r in roles if "combiner" in r)
    mzm_count = sum(
        1 for d in descriptions
        if "mach-zehnder modulator" in d or "mzm" in d
    )
    phase_shifter_90 = any("90" in d and "phase" in d for d in descriptions)
    ring_count = sum(1 for d in descriptions if "ring" in d)

    _POWERS_OF_2 = {2, 4, 8, 16, 32, 64}

    if "power splitter tree" in architectures and n_value:
        p2_fact = (
            f" The number of output ports {n_value} is a power of 2."
            if n_value in _POWERS_OF_2 else ""
        )
        tree_splitters = sum(
            1 for d in descriptions if "splitter" in d or "mmi" in d
        )
        premises.append(
            f"A 1xN power splitter tree is constructed from 1x2 binary "
            f"splitter elements. N = {n_value}. "
            f"The tree has {tree_splitters} splitters.{p2_fact}"
        )
        claims.append(
            "The binary splitter tree specification is valid."
        )
        claims.append("The binary splitter tree design is feasible.")

    if "Benes switching network" in architectures and n_value:
        switch_count = sum(
            1 for d in descriptions if "switch" in d or "mzi" in d
        )
        premises.append(
            f"An NxN Benes switching network is specified. N = {n_value}. "
            f"The network has {switch_count} switches total. "
            f"The network has exactly {n_value} external input ports and "
            f"{n_value} external output ports."
        )
        claims.append("The Benes network specification is valid.")

    if "Clements unitary mesh" in architectures and n_value:
        mzi_count = sum(1 for d in descriptions if "mzi" in d)
        premises.append(
            f"An NxN Clements unitary mesh is specified. N = {n_value}. "
            f"The mesh contains {mzi_count} MZI blocks. "
            f"The mesh has exactly {n_value} external input ports and "
            f"{n_value} external output ports."
        )
        claims.append("The Clements mesh specification is valid.")

    if "QPSK modulator" in architectures:
        has_splitter_12 = any(
            "1x2" in d and "splitter" in d for d in descriptions
        ) or any(
            "splitter" in r and any(
                (c.port_config or "") in ("1x2", "1×2")
                for c in design_intent.components if c.role and c.role.lower() == r
            )
            for r in roles
        )
        has_combiner_21 = any(
            "2x1" in d and "combiner" in d for d in descriptions
        ) or any("combiner" in r for r in roles)

        qpsk_parts = ["A QPSK modulator is specified."]
        qpsk_parts.append(f"{mzm_count} Mach-Zehnder Modulators.")
        if has_splitter_12:
            qpsk_parts.append("A 1x2 splitter.")
        if has_combiner_21:
            qpsk_parts.append("A 2x1 combiner.")
        premises.append(" ".join(qpsk_parts))

        if phase_shifter_90:
            premises.append(
                "A 90-degree phase shifter is present on one path."
            )
        else:
            premises.append(
                "There is no 90-degree phase shifter in the circuit."
            )
        premises.append(
            "The circuit has exactly 1 optical input port and "
            "1 optical output port."
        )
        claims.append("The QPSK design is complete and valid.")
        claims.append("The QPSK port configuration is valid.")

    if "Mach-Zehnder interferometer" in architectures and not any(
        a in architectures
        for a in ["QPSK modulator", "Benes switching network",
                   "Clements unitary mesh"]
    ):
        premises.append(
            "A Mach-Zehnder interferometer topology is intended. "
            "Both arms originate from the same splitter and terminate "
            "at the same combiner."
        )
        has_balanced = any(
            "balanced" in d and "unbalanced" not in d
            for d in descriptions
        ) or any(
            s.key == "delta_length" and s.value.strip() in ("0", "0.0", "0 um")
            for c in design_intent.components for s in c.specs
        )
        has_unbalanced = any("unbalanced" in d or "asymmetric" in d for d in descriptions)
        delta_specs = [
            s.value for c in design_intent.components for s in c.specs
            if "delta" in s.key.lower() or "path_length_difference" in s.key.lower()
        ]
        if has_balanced:
            premises.append(
                "A balanced MZI is specified. The path length difference is 0."
            )
            claims.append("The balanced MZI specification is valid.")
        elif has_unbalanced or delta_specs:
            dl = delta_specs[0] if delta_specs else "> 0"
            premises.append(
                f"An unbalanced MZI is specified. "
                f"The path length difference is {dl}."
            )
            claims.append("The unbalanced MZI specification is valid.")
        claims.append("The MZI interferometric function is valid.")

    if any("WDM" in a for a in architectures) and ring_count > 0:
        n_ch = n_value or ring_count
        premises.append(
            f"An N-channel WDM demultiplexer using ring resonators is "
            f"specified. N = {n_ch}. The circuit contains {ring_count} "
            f"ring resonators."
        )
        claims.append("The WDM ring count specification is valid.")
        claims.append("The WDM ring design is valid.")

        has_add_drop = any(
            "add-drop" in d or "add/drop" in d or "dual-bus" in d
            or "double-bus" in d or "dual bus" in d or "double bus" in d
            for d in descriptions
        )
        if has_add_drop:
            premises.append(
                "Each add-drop ring resonator is a double-bus ring with "
                "4 ports: input, through, add, and drop."
            )
            claims.append("The add-drop ring configuration is valid.")

    if not claims:
        claims.append("The circuit specification is valid.")

    input_text = " ".join(premises)
    output_text = " ".join(claims)
    return input_text, output_text


# ---------------------------------------------------------------------------
# Checkpoint B: Parameter serializer
# ---------------------------------------------------------------------------

_PARAM_MAP = {
    "width": "waveguide width",
    "wg_width": "waveguide width",
    "waveguide_width": "waveguide width",
    "radius": "bend radius or ring radius",
    "bend_radius": "waveguide bend radius",
    "ring_radius": "ring resonator radius",
    "gap": "directional coupler gap",
    "coupling_gap": "ring resonator coupling gap",
    "delta_length": "path length difference for an MZI",
    "path_length_difference": "path length difference for an MZI",
    "splitting_ratio": "splitting ratio",
    "split_ratio": "splitting ratio",
    "coupling": "splitting ratio",
    "length": "waveguide length",
    "taper_length": "taper length",
    "phase_shifter_length": "phase shifter length",
    "bend_angle": "bend angle",
    "taper_angle": "grating coupler taper angle",
}

_PARAM_CLAIM = {
    "waveguide width":             "The waveguide width specification is valid.",
    "waveguide bend radius":       "The waveguide bend radius specification is valid.",
    "bend radius or ring radius":  "The waveguide bend radius specification is valid.",
    "ring resonator radius":       "The ring resonator radius specification is valid.",
    "directional coupler gap":     "The directional coupler gap specification is valid.",
    "ring resonator coupling gap": "The directional coupler gap specification is valid.",
    "path length difference for an MZI": "The path length difference specification is valid.",
    "splitting ratio":             "The splitting ratio specification is valid.",
    "waveguide length":            "The length specification is valid.",
    "taper length":                "The length specification is valid.",
    "phase shifter length":        "The length specification is valid.",
    "bend angle":                  "The bend angle specification is valid.",
    "grating coupler taper angle": "The bend angle specification is valid.",
}


def serialize_parameter_claims_per_component(
    circuit_dsl: dict,
) -> list[tuple[str, str, str]]:
    """Serialize parameter claims one component at a time.

    Returns a list of (node_id, input_text, output_text) tuples — one
    entry per component that has at least one recognised parameter.
    Each entry is a self-contained AR problem small enough to avoid
    the "tooComplex" result.
    """
    result = []
    nodes = circuit_dsl.get("nodes", {})

    for node_id, node_data in nodes.items():
        params = node_data.get("params", {})
        component = node_data.get("component", "unknown")
        premises = []
        claim_set: set[str] = set()

        for param_key, param_val in params.items():
            param_key_lower = param_key.lower()
            matched_desc = None
            for key_pattern, desc in _PARAM_MAP.items():
                if key_pattern in param_key_lower:
                    matched_desc = desc
                    break
            if matched_desc is None:
                continue
            try:
                numeric_val = float(str(param_val))
            except (ValueError, TypeError):
                continue
            premises.append(
                f"Component {node_id} ({component}) has a {matched_desc} "
                f"specified: {numeric_val}."
            )
            claim = _PARAM_CLAIM.get(matched_desc)
            if claim:
                claim_set.add(claim)

        if premises:
            result.append((
                node_id,
                " ".join(premises),
                " ".join(sorted(claim_set)),
            ))

    return result


def serialize_parameter_claims(circuit_dsl: dict) -> tuple[str, str]:
    """Convert a circuit DSL dict into a combined (Input, Output) text pair.

    This is a convenience wrapper around serialize_parameter_claims_per_component
    that concatenates all components into one Input/Output pair.  Useful for
    offline tests and logging; validate_parameters() uses the per-component
    version instead to keep individual API calls simple.
    """
    per_comp = serialize_parameter_claims_per_component(circuit_dsl)
    if not per_comp:
        return "", ""
    all_inputs = " ".join(inp for _, inp, _ in per_comp)
    all_claims: set[str] = set()
    for _, _, out in per_comp:
        all_claims.update(out.split(". "))
    cleaned = set()
    for c in all_claims:
        c = c.strip().rstrip(".")
        if c:
            cleaned.add(c + ".")
    output_text = " ".join(sorted(cleaned))
    return all_inputs, output_text


# ---------------------------------------------------------------------------
# Top-level validation functions
# ---------------------------------------------------------------------------

def _findings_to_feedback(
    findings: list[dict],
    aggregate: str,
    source_phase: str,
    target_phase: str,
    severity_override: Optional[str] = None,
) -> list[PipelineFeedback]:
    """Convert parsed AR findings into PipelineFeedback objects."""
    feedback = []

    if aggregate in ("valid", "noTranslations"):
        return feedback

    if aggregate == "satisfiable":
        logger.info(
            "AR returned satisfiable for %s — claims are consistent but "
            "not fully entailed. Proceeding without blocking.",
            source_phase,
        )
        return feedback

    severity = severity_override or (
        "fundamental" if aggregate in ("invalid", "impossible") else "major"
    )

    contradicting = []
    for f in findings:
        if f["type"] in ("invalid", "impossible"):
            contradicting.extend(f.get("contradicting_rules", []))

    description_parts = [f"AR validation result: {aggregate.upper()}."]
    for f in findings:
        if f["type"] in ("invalid", "impossible", "tooComplex",
                          "translationAmbiguous"):
            if f["claims"]:
                description_parts.append(
                    f"Claims: {json.dumps(f['claims'], default=str)}"
                )
            if f["contradicting_rules"]:
                description_parts.append(
                    f"Contradicting rules: "
                    f"{json.dumps(f['contradicting_rules'], default=str)}"
                )

    feedback.append(PipelineFeedback(
        source_phase=source_phase,
        target_phase=target_phase,
        severity=severity,
        description=" ".join(description_parts),
        affected_components=["global"],
        suggested_action=(
            "Revise the design to comply with the policy constraints. "
            f"AR result: {aggregate}."
        ),
        context={
            "ar_aggregate_result": aggregate,
            "ar_findings": [
                {
                    "type": f["type"],
                    "premises": f["premises"],
                    "claims": f["claims"],
                    "contradicting_rules": f["contradicting_rules"],
                    "supporting_rules": f["supporting_rules"],
                    "untranslated_premises": f["untranslated_premises"],
                    "untranslated_claims": f["untranslated_claims"],
                }
                for f in findings
            ],
        },
    ))

    return feedback


def validate_architecture(
    design_intent: DesignIntent,
) -> list[PipelineFeedback]:
    """Checkpoint A: Validate DesignIntent architecture against Tier 6 policy.

    Returns PipelineFeedback items for invalid/impossible results.
    Returns empty list for valid, satisfiable, or if AR is unavailable.
    """
    if not _ar_enabled():
        logger.debug("AR validation disabled or not configured, skipping architecture gate.")
        return []

    input_text, output_text = serialize_architecture_claims(design_intent)
    if not input_text or not output_text:
        logger.debug("Architecture serializer produced empty text, skipping.")
        return []

    logger.info("Calling AR architecture gate...")
    response = _call_apply_guardrail(input_text, output_text)
    if response is None:
        return []

    findings = _parse_findings(response)
    if not findings:
        logger.info("AR returned no findings for architecture gate.")
        return []

    aggregate = _get_aggregate_result(findings)
    logger.info("AR architecture gate aggregate result: %s", aggregate)

    return _findings_to_feedback(
        findings=findings,
        aggregate=aggregate,
        source_phase="ar_architecture_gate",
        target_phase="interpreter",
        severity_override="fundamental",
    )


def validate_parameters(circuit_dsl: dict) -> list[PipelineFeedback]:
    """Checkpoint B: Validate resolved circuit DSL parameters against Tier 6 policy.

    Makes one AR API call per component to keep each call simple and avoid
    "tooComplex" results.  Returns PipelineFeedback items for any component
    whose parameters are invalid/impossible.  Returns empty list when all
    components are valid/satisfiable or if AR is unavailable.
    """
    if not _ar_enabled():
        logger.debug("AR validation disabled or not configured, skipping parameter gate.")
        return []

    per_comp = serialize_parameter_claims_per_component(circuit_dsl)
    if not per_comp:
        logger.debug("Parameter serializer produced no components, skipping.")
        return []

    all_feedback: list[PipelineFeedback] = []

    for node_id, input_text, output_text in per_comp:
        logger.info("Calling AR parameter gate for component %s...", node_id)
        response = _call_apply_guardrail(input_text, output_text)
        if response is None:
            continue

        findings = _parse_findings(response)
        if not findings:
            logger.info("AR returned no findings for component %s.", node_id)
            continue

        aggregate = _get_aggregate_result(findings)
        logger.info(
            "AR parameter gate result for %s: %s", node_id, aggregate
        )

        fb = _findings_to_feedback(
            findings=findings,
            aggregate=aggregate,
            source_phase="ar_parameter_gate",
            target_phase="interpreter",
            severity_override="major",
        )
        for item in fb:
            item.affected_components = [node_id]
        all_feedback.extend(fb)

    return all_feedback
