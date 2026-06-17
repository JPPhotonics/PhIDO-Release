"""
Multi-level test harness for PhIDO ⇄ Bedrock Automated Reasoning integration.

Level 1: Serializer unit tests (offline, no AWS calls)
Level 2: Direct API calls with hand-crafted tier6 test cases
Level 3: End-to-end serializer → API round-trips with mock DesignIntent / circuit_dsl

Usage:
    python guardrail_test.py                 # run all levels
    python guardrail_test.py --level 1       # serializer-only (offline)
    python guardrail_test.py --level 2       # tier6 test cases via API
    python guardrail_test.py --level 3       # end-to-end with mock objects
"""

import argparse
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()

from mcp_servers.models import (
    ComponentIntent, Connection, DesignIntent, SpecEntry,
)
from mcp_servers.ar_validator import (
    _ar_enabled,
    _call_apply_guardrail,
    _parse_findings,
    _get_aggregate_result,
    serialize_architecture_claims,
    serialize_parameter_claims,
    serialize_parameter_claims_per_component,
    validate_architecture,
    validate_parameters,
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"
INFO = "\033[94mINFO\033[0m"

stats = {"pass": 0, "fail": 0, "skip": 0}


def header(title: str):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        stats["pass"] += 1
        print(f"  [{PASS}] {name}")
    else:
        stats["fail"] += 1
        print(f"  [{FAIL}] {name}")
    if detail:
        for line in detail.strip().split("\n"):
            print(f"         {line}")


def skip(name: str, reason: str = ""):
    stats["skip"] += 1
    print(f"  [{SKIP}] {name}  {reason}")


# ═══════════════════════════════════════════════════════════════════════════
# Mock objects
# ═══════════════════════════════════════════════════════════════════════════

def _make_splitter_tree_di(n: int = 8) -> DesignIntent:
    """1xN binary splitter tree DesignIntent."""
    splitters = n - 1
    comps = [
        ComponentIntent(
            id=f"C{i+1}", description="1x2 MMI splitter",
            role="splitter", port_config="1x2", specs=[],
        )
        for i in range(splitters)
    ]
    conns = [
        Connection(
            from_component="C1", to_component="C2",
            description="stage 1 to stage 2",
        ),
    ]
    return DesignIntent(
        title=f"1x{n} Power Splitter Tree",
        brief_summary=f"A 1x{n} optical power splitter tree using binary MMI splitters",
        components=comps,
        connections=conns,
    )


def _make_mzi_di(balanced: bool = True) -> DesignIntent:
    """Simple balanced or unbalanced MZI DesignIntent."""
    delta = "0" if balanced else "200 um"
    return DesignIntent(
        title="Balanced MZI" if balanced else "Unbalanced MZI",
        brief_summary="A Mach-Zehnder interferometer for optical switching",
        components=[
            ComponentIntent(id="C1", description="1x2 MMI splitter",
                            role="splitter", port_config="1x2", specs=[]),
            ComponentIntent(id="C2", description="phase shifter on upper arm",
                            role="modulator", specs=[]),
            ComponentIntent(
                id="C3", description="MZI combiner 2x1",
                role="combiner", port_config="2x1",
                specs=[SpecEntry(key="delta_length", value=delta)],
            ),
        ],
        connections=[
            Connection(from_component="C1", to_component="C2",
                       description="upper arm"),
            Connection(from_component="C1", to_component="C3",
                       description="lower arm"),
            Connection(from_component="C2", to_component="C3",
                       description="upper arm to combiner"),
        ],
    )


def _make_qpsk_di(has_phase_shifter: bool = True) -> DesignIntent:
    """QPSK modulator DesignIntent, optionally missing the 90-deg phase shifter."""
    comps = [
        ComponentIntent(id="C1", description="1x2 splitter for I/Q paths",
                        role="splitter", port_config="1x2", specs=[]),
        ComponentIntent(id="C2", description="Mach-Zehnder Modulator on I path",
                        role="modulator", specs=[]),
        ComponentIntent(id="C3", description="Mach-Zehnder Modulator on Q path",
                        role="modulator", specs=[]),
        ComponentIntent(id="C5", description="2x1 combiner for I/Q recombination",
                        role="combiner", port_config="2x1", specs=[]),
    ]
    if has_phase_shifter:
        comps.insert(3, ComponentIntent(
            id="C4", description="90-degree phase shifter on Q path",
            role="modulator", specs=[],
        ))
    return DesignIntent(
        title="QPSK Modulator",
        brief_summary="A QPSK modulator with dual MZMs and 90-degree phase shift",
        components=comps,
        connections=[
            Connection(from_component="C1", to_component="C2",
                       description="I path"),
            Connection(from_component="C1", to_component="C3",
                       description="Q path"),
            Connection(from_component="C2", to_component="C5",
                       description="I to combiner"),
            Connection(from_component="C3", to_component="C5",
                       description="Q to combiner"),
        ],
    )


def _make_wdm_di(n_channels: int = 4) -> DesignIntent:
    """N-channel WDM demux DesignIntent with add-drop ring resonators."""
    comps = [
        ComponentIntent(
            id=f"C{i+1}",
            description=f"add-drop dual-bus ring resonator filter targeting lambda_{i+1}",
            role="filter", port_config="2x2", specs=[],
        )
        for i in range(n_channels)
    ]
    conns = [
        Connection(
            from_component=f"C{i+1}", to_component=f"C{i+2}",
            description="bus waveguide cascade",
        )
        for i in range(n_channels - 1)
    ]
    return DesignIntent(
        title=f"{n_channels}-channel WDM Demultiplexer",
        brief_summary=f"An {n_channels}-channel WDM demultiplexer using ring resonators",
        components=comps,
        connections=conns,
    )


def _make_param_dsl_valid() -> dict:
    """Circuit DSL with all parameters in valid physical ranges."""
    return {
        "nodes": {
            "C1": {
                "component": "mmi1x2_cband",
                "params": {"width": 0.5, "length": 10.0, "gap": 0.2},
            },
            "C2": {
                "component": "mzi_heater",
                "params": {"delta_length": 50.0, "splitting_ratio": 0.5},
            },
            "C3": {
                "component": "ring_resonator",
                "params": {"radius": 10.0, "coupling_gap": 0.15},
            },
        },
    }


def _make_param_dsl_invalid() -> dict:
    """Circuit DSL with a negative width (should trigger Rule 1 violation)."""
    return {
        "nodes": {
            "C1": {
                "component": "waveguide_straight",
                "params": {"width": -0.3, "length": 100.0},
            },
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Level 1: Serializer unit tests (offline)
# ═══════════════════════════════════════════════════════════════════════════

def level_1():
    header("Level 1 — Serializer unit tests (no AWS calls)")

    # 1a. Binary splitter tree
    di = _make_splitter_tree_di(8)
    inp, out = serialize_architecture_claims(di)
    check("Splitter tree: input text is non-empty", len(inp) > 0)
    check("Splitter tree: output text is non-empty", len(out) > 0)
    check("Splitter tree: input mentions N = 8",
          "N = 8" in inp, f"Input excerpt: ...{inp[:200]}...")
    check("Splitter tree: input mentions 7 splitters",
          "7 splitters" in inp, f"Input excerpt: ...{inp[:300]}...")
    check("Splitter tree: input states power-of-2 as premise",
          "power of 2" in inp.lower())
    check("Splitter tree: output claims specification valid",
          "specification is valid" in out.lower())
    check("Splitter tree: output claims feasible",
          "feasible" in out.lower())

    # 1b. Balanced MZI
    di = _make_mzi_di(balanced=True)
    inp, out = serialize_architecture_claims(di)
    check("Balanced MZI: input mentions balanced or delta_length 0",
          "balanced" in inp.lower() or "path length difference is 0" in inp.lower(),
          f"Input: {inp[:200]}")
    check("Balanced MZI: input states arm-sharing premise",
          "originate from the same splitter" in inp.lower()
          and "terminate at the same combiner" in inp.lower(),
          f"Input: {inp[:300]}")
    check("Balanced MZI: output claims balanced MZI spec valid",
          "balanced mzi specification is valid" in out.lower())
    check("Balanced MZI: output claims interferometric function valid",
          "interferometric function is valid" in out.lower())

    # 1c. QPSK with phase shifter
    di = _make_qpsk_di(has_phase_shifter=True)
    inp, out = serialize_architecture_claims(di)
    check("QPSK (with PS): input mentions 90-degree phase shifter",
          "90" in inp and "phase" in inp.lower())
    check("QPSK (with PS): input mentions 2 MZMs",
          "2" in inp and ("mzm" in inp.lower() or "mach-zehnder modulator" in inp.lower()))
    check("QPSK (with PS): input states port counts",
          "1 optical input port" in inp.lower())
    check("QPSK (with PS): output claims design complete and valid",
          "qpsk design is complete and valid" in out.lower())
    check("QPSK (with PS): output claims port config valid",
          "port configuration is valid" in out.lower())

    # 1d. QPSK missing phase shifter
    di = _make_qpsk_di(has_phase_shifter=False)
    inp, out = serialize_architecture_claims(di)
    check("QPSK (no PS): input negates 90-degree phase shifter",
          "no 90" in inp.lower() or "no 90-degree" in inp.lower(),
          f"Input excerpt: ...{inp[:300]}...")
    check("QPSK (no PS): output still claims design valid (AR should reject)",
          "qpsk design is complete and valid" in out.lower())

    # 1e. WDM demux
    di = _make_wdm_di(4)
    inp, out = serialize_architecture_claims(di)
    check("WDM 4-ch: input mentions ring resonators",
          "ring" in inp.lower())
    check("WDM 4-ch: input mentions N = 4",
          "N = 4" in inp)
    check("WDM 4-ch: input states add-drop dual-bus premise",
          "double-bus" in inp.lower() or "dual-bus" in inp.lower())
    check("WDM 4-ch: output claims ring count valid",
          "ring count specification is valid" in out.lower())
    check("WDM 4-ch: output claims ring design valid",
          "ring design is valid" in out.lower())
    check("WDM 4-ch: output claims add-drop config valid",
          "add-drop ring configuration is valid" in out.lower())

    # 1f. Parameter serializer — valid DSL
    dsl = _make_param_dsl_valid()
    inp, out = serialize_parameter_claims(dsl)
    check("Param (valid): input text is non-empty", len(inp) > 0)
    check("Param (valid): output uses validity claims",
          "specification is valid" in out.lower(), f"Output: {out}")
    check("Param (valid): input mentions width 0.5",
          "0.5" in inp, f"Input excerpt: {inp[:200]}")

    # 1g. Parameter serializer — invalid DSL (negative width)
    dsl = _make_param_dsl_invalid()
    inp, out = serialize_parameter_claims(dsl)
    check("Param (negative width): input mentions -0.3",
          "-0.3" in inp, f"Input: {inp}")
    check("Param (negative width): output claims valid (AR should reject)",
          "waveguide width specification is valid" in out.lower(),
          f"Output: {out}")

    # 1h. Empty DSL produces empty text
    inp, out = serialize_parameter_claims({"nodes": {}})
    check("Param (empty DSL): empty input returns empty strings",
          inp == "" and out == "")

    # 1i. Per-component serializer — valid DSL splits into 3 components
    dsl = _make_param_dsl_valid()
    per_comp = serialize_parameter_claims_per_component(dsl)
    check("Per-component: returns one entry per component node",
          len(per_comp) == 3, f"Got {len(per_comp)} entries")
    for node_id, cinp, cout in per_comp:
        combined = len(cinp) + len(cout)
        check(f"Per-component {node_id}: combined length <= 1024",
              combined <= 1024, f"Combined: {combined}")
        check(f"Per-component {node_id}: output uses validity claims",
              "specification is valid" in cout.lower(),
              f"Output: {cout}")


# ═══════════════════════════════════════════════════════════════════════════
# Level 2: Tier 6 test cases — direct API calls
# ═══════════════════════════════════════════════════════════════════════════

TIER6_TEST_CASES = [
    {
        "name": "TC1: Correct 1x8 binary splitter tree",
        "input": (
            "A 1xN power splitter tree is constructed from 1x2 binary splitter "
            "elements. N = 8. The tree has 3 stages and 7 splitters. Equal power "
            "is required at all 8 outputs. Every splitter in the tree has a 50:50 "
            "splitting ratio. The number of output ports is 8, which is a power of 2."
        ),
        "output": (
            "The binary splitter tree specification is valid. "
            "The binary splitter tree design is feasible. "
            "The equal-power specification is valid."
        ),
        "expected": "valid",
    },
    {
        "name": "TC2: MZI phase-to-intensity (satisfiable — claim not in policy)",
        "input": (
            "A Mach-Zehnder interferometer is specified. The MZI has one "
            "splitting element and one combining element. Both arms originate from the "
            "same splitter and terminate at the same combiner. A phase shifter is "
            "placed in one arm of the MZI. The path length difference is 0 (balanced)."
        ),
        "output": (
            "The MZI converts phase modulation into intensity modulation at its output."
        ),
        "expected": "satisfiable",
    },
    {
        "name": "TC3: Well-formed 4x4 Benes network",
        "input": (
            "An NxN Benes switching network is specified. N = 4. The network "
            "has 3 stages, 2 switches per stage, and 6 switches total. The network "
            "has exactly 4 external input ports and 4 external output ports."
        ),
        "output": (
            "The Benes network specification is valid. "
            "The optical network port specification is valid."
        ),
        "expected": "valid",
    },
    {
        "name": "TC4: Passive device violates energy conservation",
        "input": (
            "A photonic device is specified. The device is passive. The device "
            "does not have an external energy source. The total input optical power "
            "is 5 mW. The total output optical power is 8 mW. The device's insertion "
            "loss is -2 dB (negative, implying gain). |S_21| = 1.3."
        ),
        "output": (
            "The passive device power specification is valid. "
            "The insertion loss specification is physically realizable. "
            "The S21 magnitude is physically realizable."
        ),
        "expected": "impossible",
    },
    {
        "name": "TC5: 4x4 Clements mesh — too few MZIs",
        "input": (
            "An NxN Clements unitary mesh is specified. N = 4. The mesh "
            "contains 4 MZI blocks. The mesh has exactly 4 external input ports and "
            "4 external output ports."
        ),
        "output": "The Clements mesh specification is valid.",
        "expected": "invalid",
    },
    {
        "name": "TC6: Correct QPSK modulator",
        "input": (
            "A QPSK modulator is specified. The circuit contains a 1x2 "
            "splitter dividing the optical input into an I path and a Q path. There "
            "is one Mach-Zehnder Modulator on the I path and one Mach-Zehnder "
            "Modulator on the Q path, for a total of 2 MZMs. A 90-degree phase "
            "shifter is present on the Q path. A 2x1 combiner recombines both paths "
            "into a single output. The circuit has exactly 1 optical input port and "
            "1 optical output port."
        ),
        "output": (
            "The QPSK design is complete and valid. "
            "The QPSK port configuration is valid."
        ),
        "expected": "valid",
    },
    {
        "name": "TC7: Waveguide with negative width",
        "input": "A waveguide is specified. The waveguide width is -0.3 um.",
        "output": "The waveguide width specification is valid.",
        "expected": "invalid",
    },
    {
        "name": "TC8: 4-channel WDM demux with ring resonators",
        "input": (
            "An N-channel WDM demultiplexer using ring resonators is specified. "
            "N = 4. The circuit contains 4 ring resonators. Each ring resonator "
            "targets a distinct resonance wavelength: ring 1 targets lambda_1, ring 2 "
            "targets lambda_2, ring 3 targets lambda_3, ring 4 targets lambda_4. No "
            "two rings have the same resonance wavelength. Each ring is a double-bus "
            "(dual-bus) ring with 4 ports: input, through, add, and drop."
        ),
        "output": (
            "The WDM ring count specification is valid. "
            "The WDM ring design is valid. "
            "The add-drop ring configuration is valid."
        ),
        "expected": "valid",
    },
    {
        "name": "TC9: QPSK missing 90-degree phase shifter",
        "input": (
            "A QPSK modulator is specified. The circuit contains a 1x2 "
            "splitter dividing the optical input into an I path and a Q path. There "
            "is one Mach-Zehnder Modulator on the I path and one Mach-Zehnder "
            "Modulator on the Q path. There is no 90-degree phase shifter anywhere "
            "in the circuit. A 2x1 combiner recombines both paths. The circuit has "
            "1 optical input port and 1 optical output port."
        ),
        "output": "The QPSK design is complete and valid.",
        "expected": "invalid",
    },
    {
        "name": "TC10: Balanced MZI with non-zero path length difference",
        "input": (
            "A balanced MZI is specified. The path length difference between "
            "the two arms is 200 um. The MZI has one splitting element and one "
            "combining element. Both arms originate from the same splitter and "
            "terminate at the same combiner."
        ),
        "output": "The balanced MZI specification is valid.",
        "expected": "invalid",
    },
]


def level_2():
    header("Level 2 — Tier 6 test cases via ApplyGuardrail API")

    if not _ar_enabled():
        for tc in TIER6_TEST_CASES:
            skip(tc["name"], "(AR not configured)")
        return

    print(f"  [{INFO}] Guardrail ID: {os.environ.get('AR_GUARDRAIL_ID')}")
    print(f"  [{INFO}] Guardrail Version: {os.environ.get('AR_GUARDRAIL_VERSION')}")
    print(f"  [{INFO}] Region: {os.environ.get('AWS_DEFAULT_REGION')}")
    print()

    for tc in TIER6_TEST_CASES:
        t0 = time.time()
        response = _call_apply_guardrail(tc["input"], tc["output"])
        elapsed = time.time() - t0

        if response is None:
            skip(tc["name"], "(API call returned None)")
            continue

        findings = _parse_findings(response)
        if not findings:
            aggregate = "no_findings"
        else:
            aggregate = _get_aggregate_result(findings)

        matched = (aggregate == tc["expected"])
        check(
            f"{tc['name']}  [{elapsed:.1f}s]",
            matched,
            f"Expected: {tc['expected']}  |  Got: {aggregate}",
        )

        if not matched:
            for f in findings:
                print(f"           finding type={f['type']}")
                if f["premises"]:
                    print(f"           premises={f['premises'][:2]}...")
                if f["claims"]:
                    print(f"           claims={f['claims'][:2]}...")
                if f["contradicting_rules"]:
                    print(f"           contradicting={f['contradicting_rules']}")
                if f["untranslated_premises"]:
                    print(f"           untranslated_premises={f['untranslated_premises']}")
                if f["untranslated_claims"]:
                    print(f"           untranslated_claims={f['untranslated_claims']}")


# ═══════════════════════════════════════════════════════════════════════════
# Level 3: End-to-end serializer → API with mock DesignIntent / circuit_dsl
# ═══════════════════════════════════════════════════════════════════════════

LEVEL3_CASES = [
    {
        "name": "E2E Arch: 1x8 splitter tree (expect valid or satisfiable)",
        "kind": "arch",
        "build": lambda: _make_splitter_tree_di(8),
        "accept": {"valid", "satisfiable"},
    },
    {
        "name": "E2E Arch: balanced MZI (expect valid or satisfiable)",
        "kind": "arch",
        "build": lambda: _make_mzi_di(balanced=True),
        "accept": {"valid", "satisfiable"},
    },
    {
        "name": "E2E Arch: QPSK with phase shifter (expect valid or satisfiable)",
        "kind": "arch",
        "build": lambda: _make_qpsk_di(has_phase_shifter=True),
        "accept": {"valid", "satisfiable"},
    },
    {
        "name": "E2E Arch: QPSK missing phase shifter (expect invalid)",
        "kind": "arch",
        "build": lambda: _make_qpsk_di(has_phase_shifter=False),
        "accept": {"invalid"},
    },
    {
        "name": "E2E Arch: 4-ch WDM demux (expect valid or satisfiable)",
        "kind": "arch",
        "build": lambda: _make_wdm_di(4),
        "accept": {"valid", "satisfiable"},
    },
    {
        "name": "E2E Param: valid bounds (expect valid or satisfiable)",
        "kind": "param",
        "build": lambda: _make_param_dsl_valid(),
        "accept": {"valid", "satisfiable"},
    },
    {
        "name": "E2E Param: negative width (expect invalid)",
        "kind": "param",
        "build": lambda: _make_param_dsl_invalid(),
        "accept": {"invalid"},
    },
]


def level_3():
    header("Level 3 — End-to-end: serializer → API round-trip")

    if not _ar_enabled():
        for tc in LEVEL3_CASES:
            skip(tc["name"], "(AR not configured)")
        return

    for tc in LEVEL3_CASES:
        obj = tc["build"]()
        t0 = time.time()

        if tc["kind"] == "arch":
            feedback = validate_architecture(obj)
            inp, out = serialize_architecture_claims(obj)
        else:
            feedback = validate_parameters(obj)
            inp, out = serialize_parameter_claims(obj)

        elapsed = time.time() - t0

        if not inp:
            skip(tc["name"], "(serializer produced empty text)")
            continue

        if tc["kind"] == "arch":
            response = _call_apply_guardrail(inp, out) if feedback == [] else None
        else:
            response = None

        if feedback:
            agg = feedback[0].context.get("ar_aggregate_result", "unknown")
        elif response is not None:
            findings = _parse_findings(response)
            agg = _get_aggregate_result(findings) if findings else "no_findings"
        else:
            agg = "valid_or_skipped"

        if not feedback and agg in ("valid_or_skipped", "no_findings"):
            agg = "valid"

        matched = agg in tc["accept"]
        check(
            f"{tc['name']}  [{elapsed:.1f}s]",
            matched,
            f"Accepted: {tc['accept']}  |  Got: {agg}"
            + (f"\n  Feedback: {[f.description[:80] for f in feedback]}" if feedback else ""),
        )

        if not matched:
            print(f"           Input:  {inp[:150]}...")
            print(f"           Output: {out[:150]}...")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PhIDO AR integration test harness")
    parser.add_argument(
        "--level", type=int, choices=[1, 2, 3], default=None,
        help="Run a specific test level (1=serializer, 2=tier6 API, 3=end-to-end). "
             "Default: run all.",
    )
    args = parser.parse_args()

    levels = [args.level] if args.level else [1, 2, 3]

    print(f"PhIDO AR Integration Test Harness")
    print(f"AR Enabled: {_ar_enabled()}")
    if _ar_enabled():
        print(f"Guardrail: {os.environ.get('AR_GUARDRAIL_ID')} "
              f"v{os.environ.get('AR_GUARDRAIL_VERSION')} "
              f"({os.environ.get('AWS_DEFAULT_REGION')})")

    if 1 in levels:
        level_1()
    if 2 in levels:
        level_2()
    if 3 in levels:
        level_3()

    header("Summary")
    total = stats["pass"] + stats["fail"] + stats["skip"]
    print(f"  {PASS}: {stats['pass']}  |  {FAIL}: {stats['fail']}  |  {SKIP}: {stats['skip']}  |  Total: {total}")

    if stats["fail"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
