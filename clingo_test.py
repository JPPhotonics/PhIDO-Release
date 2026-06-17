"""
Multi-level test harness for Clingo topology validation.

Level 1: Fact generation unit tests (offline, no Clingo needed)
Level 2: Clingo solver unit tests (requires clingo)
Level 3: End-to-end DesignIntent → validate_topology() (requires clingo)

Usage:
    python clingo_test.py                 # run all levels
    python clingo_test.py --level 1       # fact generation only
    python clingo_test.py --level 2       # solver tests
    python clingo_test.py --level 3       # end-to-end
"""

import argparse
import sys

from mcp_servers.models import (
    ComponentIntent, Connection, DesignIntent, SpecEntry,
    UserRequirement, RequirementManifest, RequirementTrace,
)
from mcp_servers.clingo_validator import (
    design_intent_to_facts,
    run_clingo,
    validate_topology,
    _CLINGO_AVAILABLE,
    RULES_PATH,
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


def check(label: str, condition: bool, detail: str = ""):
    if condition:
        stats["pass"] += 1
        print(f"  [{PASS}] {label}")
    else:
        stats["fail"] += 1
        msg = f"  [{FAIL}] {label}"
        if detail:
            msg += f"  — {detail}"
        print(msg)


# ═══════════════════════════════════════════════════════════════════════════
# Mock builders (enriched DesignIntent objects)
# ═══════════════════════════════════════════════════════════════════════════

def _make_splitter_tree(n: int = 8) -> DesignIntent:
    splitters = n - 1
    comps = [
        ComponentIntent(
            id=f"C{i+1}", description="1x2 MMI splitter",
            role="splitter", port_config="1x2",
            component_type="splitter", sub_type="mmi",
        )
        for i in range(splitters)
    ]
    conns = [
        Connection(from_component=f"C{i+1}", to_component=f"C{i+2}",
                   description="stage connection")
        for i in range(min(splitters - 1, 3))
    ]
    return DesignIntent(
        title=f"1x{n} Power Splitter Tree",
        brief_summary=f"A 1x{n} optical power splitter tree using binary MMI splitters",
        components=comps, connections=conns,
        architecture_type="splitter_tree", n_value=n,
    )


def _make_mzi(balanced: bool = True) -> DesignIntent:
    delta = "0" if balanced else "200 um"
    return DesignIntent(
        title="Balanced MZI" if balanced else "Unbalanced MZI",
        brief_summary="A Mach-Zehnder interferometer for optical switching",
        components=[
            ComponentIntent(id="C1", description="1x2 MMI splitter",
                            role="splitter", port_config="1x2",
                            component_type="splitter", sub_type="mmi"),
            ComponentIntent(id="C2", description="phase shifter on upper arm",
                            role="modulator",
                            component_type="phase_shifter", sub_type="heater"),
            ComponentIntent(id="C3", description="MZI combiner 2x1",
                            role="combiner", port_config="2x1",
                            component_type="combiner", sub_type="mmi",
                            specs=[SpecEntry(key="delta_length", value=delta)]),
        ],
        connections=[
            Connection(from_component="C1", to_component="C2", description="upper arm"),
            Connection(from_component="C1", to_component="C3", description="lower arm"),
            Connection(from_component="C2", to_component="C3", description="upper arm to combiner"),
        ],
        architecture_type="mzi",
    )


def _make_qpsk(has_phase_shifter: bool = True) -> DesignIntent:
    comps = [
        ComponentIntent(id="C1", description="1x2 splitter for I/Q paths",
                        role="splitter", port_config="1x2",
                        component_type="splitter", sub_type="mmi"),
        ComponentIntent(id="C2", description="Mach-Zehnder Modulator on I path",
                        role="modulator",
                        component_type="mzm"),
        ComponentIntent(id="C3", description="Mach-Zehnder Modulator on Q path",
                        role="modulator",
                        component_type="mzm"),
        ComponentIntent(id="C5", description="2x1 combiner for I/Q recombination",
                        role="combiner", port_config="2x1",
                        component_type="combiner", sub_type="mmi"),
    ]
    if has_phase_shifter:
        comps.insert(3, ComponentIntent(
            id="C4", description="90-degree phase shifter on Q path",
            role="modulator",
            component_type="phase_shifter", sub_type="90_degree",
        ))
    conns = [
        Connection(from_component="C1", to_component="C2", description="I path"),
        Connection(from_component="C1", to_component="C3", description="Q path"),
        Connection(from_component="C2", to_component="C5", description="I to combiner"),
        Connection(from_component="C3", to_component="C5", description="Q to combiner"),
    ]
    return DesignIntent(
        title="QPSK Modulator",
        brief_summary="A QPSK modulator with dual MZMs and 90-degree phase shift",
        components=comps, connections=conns,
        architecture_type="qpsk",
    )


def _make_wdm(n_channels: int = 4) -> DesignIntent:
    comps = [
        ComponentIntent(
            id=f"C{i+1}",
            description=f"add-drop dual-bus ring resonator filter targeting lambda_{i+1}",
            role="filter", port_config="2x2",
            component_type="ring_resonator", sub_type="add_drop",
        )
        for i in range(n_channels)
    ]
    conns = [
        Connection(from_component=f"C{i+1}", to_component=f"C{i+2}",
                   description="bus waveguide cascade")
        for i in range(n_channels - 1)
    ]
    return DesignIntent(
        title=f"{n_channels}-channel WDM Demultiplexer",
        brief_summary=f"An {n_channels}-channel WDM demultiplexer using ring resonators",
        components=comps, connections=conns,
        architecture_type="wdm_demux", n_value=n_channels,
    )


def _make_with_requirements(base_di: DesignIntent, n_reqs: int = 3,
                            traced: int = 2) -> DesignIntent:
    """Wrap a base DesignIntent with a requirement manifest and partial traces."""
    reqs = [
        UserRequirement(
            id=f"R{i+1}", category="structural",
            description=f"Test requirement {i+1}",
            source_span=f"requirement {i+1} from prompt",
            priority="explicit",
        )
        for i in range(n_reqs)
    ]
    manifest = RequirementManifest(requirements=reqs, original_prompt="test prompt")

    traces = [
        RequirementTrace(
            requirement_id=f"R{i+1}",
            satisfied_by=["C1"],
            satisfaction_type="direct",
        )
        for i in range(traced)
    ]

    return base_di.model_copy(update={
        "requirement_manifest": manifest,
        "requirement_traces": traces,
    })


def _make_fallback_splitter_tree(n: int = 8) -> DesignIntent:
    """Splitter tree WITHOUT enriched fields — tests fallback detection."""
    splitters = n - 1
    comps = [
        ComponentIntent(
            id=f"C{i+1}", description="1x2 MMI splitter",
            role="splitter", port_config="1x2",
        )
        for i in range(splitters)
    ]
    return DesignIntent(
        title=f"1x{n} Power Splitter Tree",
        brief_summary=f"A 1x{n} optical power splitter tree using binary MMI splitters",
        components=comps,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Level 1: Fact generation unit tests (offline)
# ═══════════════════════════════════════════════════════════════════════════

def level_1():
    header("Level 1 — Fact generation unit tests (no Clingo needed)")

    # --- Splitter tree ---
    print("\n  [Test] Splitter tree facts (N=8)")
    di = _make_splitter_tree(8)
    facts = design_intent_to_facts(di)
    check("architecture(splitter_tree) present",
          "architecture(splitter_tree)." in facts)
    check("n_value(8) present", "n_value(8)." in facts)
    splitter_atoms = [l for l in facts.split("\n") if "component(" in l and "splitter" in l]
    check(f"7 splitter components (found {len(splitter_atoms)})",
          len(splitter_atoms) == 7)

    # --- MZI ---
    print("\n  [Test] MZI facts")
    di = _make_mzi()
    facts = design_intent_to_facts(di)
    check("architecture(mzi) present", "architecture(mzi)." in facts)
    check("splitter component present",
          any("splitter" in l and "component(" in l for l in facts.split("\n")))
    check("combiner component present",
          any("combiner" in l and "component(" in l for l in facts.split("\n")))
    check("connection atoms present",
          any("connection(" in l for l in facts.split("\n")))

    # --- QPSK ---
    print("\n  [Test] QPSK facts")
    di = _make_qpsk()
    facts = design_intent_to_facts(di)
    check("architecture(qpsk) present", "architecture(qpsk)." in facts)
    mzm_atoms = [l for l in facts.split("\n") if "component(" in l and ", mzm," in l]
    check(f"2 MZM components (found {len(mzm_atoms)})", len(mzm_atoms) == 2)
    check("90_degree sub_type present", '"90_degree"' in facts)

    # --- WDM ---
    print("\n  [Test] WDM facts (N=4)")
    di = _make_wdm(4)
    facts = design_intent_to_facts(di)
    check("architecture(wdm_demux) present", "architecture(wdm_demux)." in facts)
    check("n_value(4) present", "n_value(4)." in facts)
    ring_atoms = [l for l in facts.split("\n") if "component(" in l and "ring_resonator" in l]
    check(f"4 ring components (found {len(ring_atoms)})", len(ring_atoms) == 4)

    # --- Requirement facts ---
    print("\n  [Test] Requirement facts")
    di = _make_with_requirements(_make_mzi(), n_reqs=3, traced=2)
    facts = design_intent_to_facts(di)
    req_atoms = [l for l in facts.split("\n") if l.startswith("requirement(")]
    trace_atoms = [l for l in facts.split("\n") if l.startswith("requirement_trace(")]
    check(f"3 requirement atoms (found {len(req_atoms)})", len(req_atoms) == 3)
    check(f"2 trace atoms (found {len(trace_atoms)})", len(trace_atoms) == 2)
    check("trace includes 'direct'", any("direct" in t for t in trace_atoms))

    # --- Fallback detection ---
    print("\n  [Test] Fallback architecture detection")
    di = _make_fallback_splitter_tree(8)
    facts = design_intent_to_facts(di)
    check("fallback: architecture(splitter_tree) detected",
          "architecture(splitter_tree)." in facts)
    check("fallback: n_value(8) detected", "n_value(8)." in facts)
    splitter_atoms = [l for l in facts.split("\n") if "component(" in l and "splitter" in l]
    check(f"fallback: 7 splitter components (found {len(splitter_atoms)})",
          len(splitter_atoms) == 7)


# ═══════════════════════════════════════════════════════════════════════════
# Level 2: Clingo solver unit tests
# ═══════════════════════════════════════════════════════════════════════════

def level_2():
    header("Level 2 — Clingo solver unit tests")

    if not _CLINGO_AVAILABLE:
        print(f"  [{SKIP}] Clingo not installed, skipping Level 2.")
        stats["skip"] += 1
        return

    rules = str(RULES_PATH)

    # --- Valid 1x8 tree ---
    print("\n  [Test] Valid 1x8 splitter tree")
    facts = (
        "architecture(splitter_tree). n_value(8).\n"
        + "\n".join(f'component("c{i}", splitter, "mmi").' for i in range(1, 8))
    )
    r = run_clingo(facts, rules)
    check("satisfiable", r.satisfiable)
    check(f"no errors (found {len(r.errors)})", len(r.errors) == 0,
          str(r.answer_set))

    # --- Invalid tree N=7 ---
    print("\n  [Test] Invalid tree N=7 (not power of 2)")
    facts = (
        "architecture(splitter_tree). n_value(7).\n"
        + "\n".join(f'component("c{i}", splitter, "mmi").' for i in range(1, 7))
    )
    r = run_clingo(facts, rules)
    error_codes = [e["code"] for e in r.errors]
    check("tree_not_power_of_2 error",
          "tree_not_power_of_2" in error_codes, str(error_codes))

    # --- Wrong splitter count ---
    print("\n  [Test] Wrong splitter count (5 for N=8)")
    facts = (
        "architecture(splitter_tree). n_value(8).\n"
        + "\n".join(f'component("c{i}", splitter, "mmi").' for i in range(1, 6))
    )
    r = run_clingo(facts, rules)
    error_codes = [e["code"] for e in r.errors]
    check("tree_wrong_splitters error",
          "tree_wrong_splitters" in error_codes, str(error_codes))

    # --- Valid MZI ---
    print("\n  [Test] Valid MZI")
    facts = (
        'architecture(mzi).\n'
        'component("c1", splitter, "mmi").\n'
        'component("c2", phase_shifter, "heater").\n'
        'component("c3", combiner, "mmi").\n'
        'connection("c1", "c2"). connection("c2", "c3").\n'
        'connection("c1", "c3").\n'
    )
    r = run_clingo(facts, rules)
    check("satisfiable", r.satisfiable)
    check(f"no errors (found {len(r.errors)})", len(r.errors) == 0,
          str(r.answer_set))

    # --- MZI missing combiner ---
    print("\n  [Test] MZI missing combiner")
    facts = (
        'architecture(mzi).\n'
        'component("c1", splitter, "mmi").\n'
        'component("c2", phase_shifter, "heater").\n'
    )
    r = run_clingo(facts, rules)
    error_codes = [e["code"] for e in r.errors]
    check("mzi_wrong_combiner_count error",
          "mzi_wrong_combiner_count" in error_codes, str(error_codes))

    # --- Valid QPSK ---
    print("\n  [Test] Valid QPSK")
    facts = (
        'architecture(qpsk).\n'
        'component("c1", splitter, "mmi").\n'
        'component("c2", mzm, "none").\n'
        'component("c3", mzm, "none").\n'
        'component("c4", phase_shifter, "90_degree").\n'
        'component("c5", combiner, "mmi").\n'
    )
    r = run_clingo(facts, rules)
    check("satisfiable", r.satisfiable)
    check(f"no errors (found {len(r.errors)})", len(r.errors) == 0,
          str(r.answer_set))

    # --- QPSK missing 90-deg phase shifter ---
    print("\n  [Test] QPSK missing 90-deg phase shifter")
    facts = (
        'architecture(qpsk).\n'
        'component("c1", splitter, "mmi").\n'
        'component("c2", mzm, "none").\n'
        'component("c3", mzm, "none").\n'
        'component("c5", combiner, "mmi").\n'
    )
    r = run_clingo(facts, rules)
    error_codes = [e["code"] for e in r.errors]
    check("qpsk_missing_90ps error",
          "qpsk_missing_90ps" in error_codes, str(error_codes))

    # --- Valid 4-channel WDM ---
    print("\n  [Test] Valid 4-channel WDM")
    facts = (
        "architecture(wdm_demux). n_value(4).\n"
        + "\n".join(f'component("c{i}", ring_resonator, "add_drop").' for i in range(1, 5))
    )
    r = run_clingo(facts, rules)
    check("satisfiable", r.satisfiable)
    check(f"no errors (found {len(r.errors)})", len(r.errors) == 0,
          str(r.answer_set))

    # --- WDM insufficient rings ---
    print("\n  [Test] WDM 3 rings for 4-channel")
    facts = (
        "architecture(wdm_demux). n_value(4).\n"
        + "\n".join(f'component("c{i}", ring_resonator, "add_drop").' for i in range(1, 4))
    )
    r = run_clingo(facts, rules)
    error_codes = [e["code"] for e in r.errors]
    check("wdm_insufficient_rings error",
          "wdm_insufficient_rings" in error_codes, str(error_codes))

    # --- Unsatisfied requirement ---
    print("\n  [Test] Unsatisfied requirement")
    facts = (
        'architecture(mzi).\n'
        'component("c1", splitter, "mmi").\n'
        'component("c2", combiner, "mmi").\n'
        'connection("c1", "c2").\n'
        'requirement("r1"). requirement("r2").\n'
        'requirement_trace("r1", direct).\n'
    )
    r = run_clingo(facts, rules)
    error_codes = [e["code"] for e in r.errors]
    check("unsatisfied_requirement error",
          "unsatisfied_requirement" in error_codes, str(error_codes))


# ═══════════════════════════════════════════════════════════════════════════
# Level 3: End-to-end DesignIntent → validate_topology()
# ═══════════════════════════════════════════════════════════════════════════

def level_3():
    header("Level 3 — End-to-end DesignIntent → validate_topology()")

    if not _CLINGO_AVAILABLE:
        print(f"  [{SKIP}] Clingo not installed, skipping Level 3.")
        stats["skip"] += 1
        return

    # --- Valid splitter tree ---
    print("\n  [Test] E2E: Valid 1x8 splitter tree")
    di = _make_splitter_tree(8)
    fb = validate_topology(di)
    check(f"no feedback (got {len(fb)})", len(fb) == 0,
          "; ".join(f.description for f in fb))

    # --- Invalid splitter tree (wrong count) ---
    print("\n  [Test] E2E: Splitter tree with wrong N")
    di = _make_splitter_tree(7)
    fb = validate_topology(di)
    check("feedback returned", len(fb) > 0)
    if fb:
        codes = [f.context.get("clingo_error_code") for f in fb]
        check("tree_not_power_of_2 in feedback",
              "tree_not_power_of_2" in codes, str(codes))

    # --- Valid MZI ---
    print("\n  [Test] E2E: Valid balanced MZI")
    di = _make_mzi(balanced=True)
    fb = validate_topology(di)
    check(f"no feedback (got {len(fb)})", len(fb) == 0,
          "; ".join(f.description for f in fb))

    # --- Valid QPSK ---
    print("\n  [Test] E2E: Valid QPSK")
    di = _make_qpsk(has_phase_shifter=True)
    fb = validate_topology(di)
    check(f"no feedback (got {len(fb)})", len(fb) == 0,
          "; ".join(f.description for f in fb))

    # --- QPSK missing phase shifter ---
    print("\n  [Test] E2E: QPSK missing 90-deg PS")
    di = _make_qpsk(has_phase_shifter=False)
    fb = validate_topology(di)
    check("feedback returned", len(fb) > 0)
    if fb:
        codes = [f.context.get("clingo_error_code") for f in fb]
        check("qpsk_missing_90ps in feedback",
              "qpsk_missing_90ps" in codes, str(codes))

    # --- Valid WDM ---
    print("\n  [Test] E2E: Valid 4-channel WDM")
    di = _make_wdm(4)
    fb = validate_topology(di)
    check(f"no feedback (got {len(fb)})", len(fb) == 0,
          "; ".join(f.description for f in fb))

    # --- WDM insufficient rings ---
    print("\n  [Test] E2E: WDM 3 rings for 4 channels")
    di = _make_wdm(4)
    di.components.pop()  # remove one ring
    fb = validate_topology(di)
    check("feedback returned", len(fb) > 0)

    # --- Requirement coverage ---
    print("\n  [Test] E2E: Unsatisfied requirement")
    di = _make_with_requirements(_make_mzi(), n_reqs=3, traced=2)
    fb = validate_topology(di)
    check("feedback for unsatisfied R3", len(fb) > 0)
    if fb:
        codes = [f.context.get("clingo_error_code") for f in fb]
        check("unsatisfied_requirement in feedback",
              "unsatisfied_requirement" in codes, str(codes))

    # --- All requirements satisfied ---
    print("\n  [Test] E2E: All requirements satisfied")
    di = _make_with_requirements(_make_mzi(), n_reqs=2, traced=2)
    fb = validate_topology(di)
    req_errors = [f for f in fb if "requirement" in f.context.get("clingo_error_code", "")]
    check(f"no requirement errors (got {len(req_errors)})", len(req_errors) == 0)

    # --- Fallback detection end-to-end ---
    print("\n  [Test] E2E: Fallback splitter tree (no enriched fields)")
    di = _make_fallback_splitter_tree(8)
    fb = validate_topology(di)
    check(f"no feedback (got {len(fb)})", len(fb) == 0,
          "; ".join(f.description for f in fb))


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Clingo topology validation tests")
    parser.add_argument("--level", type=int, choices=[1, 2, 3],
                        help="Run only this test level")
    args = parser.parse_args()

    levels = [args.level] if args.level else [1, 2, 3]

    print(f"\nClingo available: {_CLINGO_AVAILABLE}")
    print(f"Rules path: {RULES_PATH}")
    print(f"Rules file exists: {RULES_PATH.exists()}")

    if 1 in levels:
        level_1()
    if 2 in levels:
        level_2()
    if 3 in levels:
        level_3()

    header("Summary")
    total = stats["pass"] + stats["fail"] + stats["skip"]
    print(f"  Total: {total}  |  {PASS}: {stats['pass']}  |  "
          f"{FAIL}: {stats['fail']}  |  {SKIP}: {stats['skip']}")
    if stats["fail"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
