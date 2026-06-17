"""Tests for mcp_servers.circuit_graph.CircuitGraph."""

import pytest

from mcp_servers.circuit_graph import CircuitGraph, PDKEntry, _make_ports


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _graph_with_two_components() -> CircuitGraph:
    """Return a graph with a 1x2 splitter at (0,0) and a 2x2 MZM at (1,0)."""
    g = CircuitGraph()
    g.add_component("splitter", "1x2", stage=0, lane=0, role="splitter",
                     description="1x2 MMI splitter")
    g.add_component("mzm", "2x2", stage=1, lane=0, role="modulator",
                     description="MZI modulator")
    return g


# ===== _make_ports =========================================================

class TestMakePorts:
    def test_1x2(self):
        ports = _make_ports("1x2")
        assert set(ports.keys()) == {"o1", "o2", "o3"}
        assert ports["o1"].direction == "input"
        assert ports["o2"].direction == "output"
        assert ports["o3"].direction == "output"

    def test_2x2(self):
        ports = _make_ports("2x2")
        assert set(ports.keys()) == {"o1", "o2", "o3", "o4"}
        assert ports["o1"].direction == "input"
        assert ports["o2"].direction == "input"
        assert ports["o3"].direction == "output"
        assert ports["o4"].direction == "output"

    def test_1x1(self):
        ports = _make_ports("1x1")
        assert set(ports.keys()) == {"o1", "o2"}
        assert ports["o1"].direction == "input"
        assert ports["o2"].direction == "output"

    def test_invalid(self):
        with pytest.raises(ValueError, match="Invalid port_config"):
            _make_ports("abc")


# ===== add_component =======================================================

class TestAddComponent:
    def test_valid_add(self):
        g = CircuitGraph()
        res = g.add_component("splitter", "1x2", stage=0, lane=0)
        assert res["status"] == "ok"
        assert res["id"] == "C1"
        assert res["position"] == {"stage": 0, "lane": 0}
        assert "o1" in res["ports"]
        assert "o2" in res["ports"]
        assert "o3" in res["ports"]

    def test_auto_increment_id(self):
        g = CircuitGraph()
        r1 = g.add_component("splitter", "1x2", stage=0, lane=0)
        r2 = g.add_component("mzm", "2x2", stage=1, lane=0)
        assert r1["id"] == "C1"
        assert r2["id"] == "C2"

    def test_position_uniqueness(self):
        g = CircuitGraph()
        g.add_component("splitter", "1x2", stage=0, lane=0)
        res = g.add_component("mzm", "2x2", stage=0, lane=0)
        assert res["status"] == "error"
        assert "already occupied" in res["message"]

    def test_bad_port_config(self):
        g = CircuitGraph()
        res = g.add_component("splitter", "abc", stage=0, lane=0)
        assert res["status"] == "error"
        assert "Invalid port_config" in res["message"]

    def test_different_lanes_same_stage(self):
        g = CircuitGraph()
        r1 = g.add_component("mzm", "2x2", stage=0, lane=0)
        r2 = g.add_component("mzm", "2x2", stage=0, lane=1)
        assert r1["status"] == "ok"
        assert r2["status"] == "ok"


# ===== connect =============================================================

class TestConnect:
    def test_valid_connection(self):
        g = _graph_with_two_components()
        res = g.connect("C1", "o2", "C2", "o1", description="splitter to MZI")
        assert res["status"] == "connected"
        assert "C1.o2 -> C2.o1" in res["edge"]

    def test_nonexistent_component(self):
        g = _graph_with_two_components()
        res = g.connect("C99", "o1", "C2", "o1")
        assert res["status"] == "error"
        assert "does not exist" in res["message"]

    def test_nonexistent_port(self):
        g = _graph_with_two_components()
        res = g.connect("C1", "o9", "C2", "o1")
        assert res["status"] == "error"
        assert "does not exist on C1" in res["message"]
        assert "available:" in res["message"]

    def test_double_connection_source(self):
        g = _graph_with_two_components()
        g.connect("C1", "o2", "C2", "o1")
        g.add_component("mzm", "2x2", stage=1, lane=1)
        res = g.connect("C1", "o2", "C3", "o1")
        assert res["status"] == "error"
        assert "already connected" in res["message"]

    def test_double_connection_target(self):
        g = _graph_with_two_components()
        g.connect("C1", "o2", "C2", "o1")
        g.add_component("splitter", "1x2", stage=0, lane=1)
        res = g.connect("C3", "o2", "C2", "o1")
        assert res["status"] == "error"
        assert "already connected" in res["message"]

    def test_forward_flow_violation(self):
        g = CircuitGraph()
        g.add_component("mzm", "2x2", stage=2, lane=0)
        g.add_component("mzm", "2x2", stage=0, lane=0)
        res = g.connect("C1", "o3", "C2", "o1")
        assert res["status"] == "error"
        assert "Backward connection" in res["message"]

    def test_forward_flow_feedback_override(self):
        g = CircuitGraph()
        g.add_component("mzm", "2x2", stage=2, lane=0)
        g.add_component("mzm", "2x2", stage=0, lane=0)
        res = g.connect("C1", "o3", "C2", "o1", feedback=True)
        assert res["status"] == "connected"

    def test_stage_adjacency_violation(self):
        g = CircuitGraph()
        g.add_component("splitter", "1x2", stage=0, lane=0)
        g.add_component("mzm", "2x2", stage=5, lane=0)
        res = g.connect("C1", "o2", "C2", "o1")
        assert res["status"] == "error"
        assert "spans" in res["message"]

    def test_stage_adjacency_skip_override(self):
        g = CircuitGraph()
        g.add_component("splitter", "1x2", stage=0, lane=0)
        g.add_component("mzm", "2x2", stage=5, lane=0)
        res = g.connect("C1", "o2", "C2", "o1", skip=True)
        assert res["status"] == "connected"

    def test_same_stage_connection(self):
        g = CircuitGraph()
        g.add_component("mzm", "2x2", stage=0, lane=0)
        g.add_component("mzm", "2x2", stage=0, lane=1)
        res = g.connect("C1", "o3", "C2", "o1")
        assert res["status"] == "connected"

    def test_lane_proximity_warning(self):
        g = CircuitGraph()
        g.add_component("mzm", "2x2", stage=0, lane=0)
        g.add_component("mzm", "2x2", stage=1, lane=10)
        res = g.connect("C1", "o3", "C2", "o1")
        assert res["status"] == "connected"
        assert "warnings" in res
        assert any("lanes" in w for w in res["warnings"])


# ===== get_open_ports ======================================================

class TestGetOpenPorts:
    def test_all_open_initially(self):
        g = CircuitGraph()
        g.add_component("splitter", "1x2", stage=0, lane=0)
        result = g.get_open_ports()
        assert result["total_components"] == 1
        ports = result["open_ports_by_stage"]["0"]
        assert len(ports) == 3  # o1, o2, o3 all open

    def test_frontier_after_connect(self):
        g = _graph_with_two_components()
        g.connect("C1", "o2", "C2", "o1")
        result = g.get_open_ports()
        # C1: o1 (open input), o3 (open output)
        # C2: o2 (open input), o3 (open output), o4 (open output)
        all_open = []
        for stage_ports in result["open_ports_by_stage"].values():
            all_open.extend(stage_ports)
        assert len(all_open) == 5

    def test_empty_graph(self):
        g = CircuitGraph()
        result = g.get_open_ports()
        assert result["total_components"] == 0
        assert result["summary"] == "No open ports"


# ===== get_state ===========================================================

class TestGetState:
    def test_basic_state(self):
        g = _graph_with_two_components()
        state = g.get_state()
        assert state["total_components"] == 2
        assert state["total_connections"] == 0
        assert "splitter" in state["component_counts"]
        assert "mzm" in state["component_counts"]

    def test_grid_layout(self):
        g = _graph_with_two_components()
        state = g.get_state()
        assert "0" in state["grid"]
        assert "0" in state["grid"]["0"]
        assert state["grid"]["0"]["0"]["id"] == "C1"


# ===== finalize ============================================================

class TestFinalize:
    def test_clean_finalize(self):
        g = _graph_with_two_components()
        g.connect("C1", "o2", "C2", "o1")
        res = g.finalize("Test", "A test circuit", "mzi")
        assert res["design_intent_ready"] is True
        assert res["total_components"] == 2
        assert res["total_connections"] == 1

    def test_disconnected_component_warning(self):
        g = CircuitGraph()
        g.add_component("splitter", "1x2", stage=0, lane=0)
        g.add_component("mzm", "2x2", stage=1, lane=0)
        res = g.finalize("Test", "Test")
        assert res["status"] == "warning"
        assert any("no connections" in w for w in res["warnings"])

    def test_disconnected_subgraph_warning(self):
        g = CircuitGraph()
        g.add_component("splitter", "1x2", stage=0, lane=0)
        g.add_component("mzm", "2x2", stage=1, lane=0)
        g.connect("C1", "o2", "C2", "o1")
        g.add_component("mzm", "2x2", stage=0, lane=1)
        g.add_component("mzm", "2x2", stage=1, lane=1)
        g.connect("C3", "o3", "C4", "o1")
        res = g.finalize("Test", "Test")
        assert any("Disconnected subgraph" in w for w in res["warnings"])

    def test_single_component_no_warning(self):
        g = CircuitGraph()
        g.add_component("splitter", "1x2", stage=0, lane=0)
        res = g.finalize("Test", "Test")
        assert res["status"] == "ok" or not any("no connections" in w for w in res["warnings"])


# ===== to_design_intent ====================================================

class TestToDesignIntent:
    def test_round_trip(self):
        g = _graph_with_two_components()
        g.connect("C1", "o2", "C2", "o1", description="split to mod")
        g.finalize("MZI", "A simple MZI", "mzi", n_value=2)
        di = g.to_design_intent()
        assert di.title == "MZI"
        assert di.architecture_type == "mzi"
        assert di.n_value == 2
        assert len(di.components) == 2
        assert len(di.connections) == 1
        assert di.components[0].id == "C1"
        assert di.components[0].port_config == "1x2"
        assert di.components[0].role == "splitter"
        assert di.components[0].confidence == 1.0
        assert di.connections[0].from_component == "C1"
        assert di.connections[0].to_component == "C2"

    def test_specs_preserved(self):
        g = CircuitGraph()
        g.add_component("ring_resonator", "2x2", stage=0, lane=0,
                         role="filter", specs={"channel": "1", "fsr": "25nm"})
        g.finalize("Ring", "Ring filter")
        di = g.to_design_intent()
        spec_keys = {s.key for s in di.components[0].specs}
        assert "channel" in spec_keys
        assert "fsr" in spec_keys


# ===== to_dot ==============================================================

class TestToDot:
    def test_rank_constraints(self):
        g = _graph_with_two_components()
        dot = g.to_dot(highlight=False)
        assert "rank=same" in dot
        assert "C1" in dot
        assert "C2" in dot

    def test_highlight_new_node(self):
        g = CircuitGraph()
        g.add_component("splitter", "1x2", stage=0, lane=0)
        dot = g.to_dot(highlight=True)
        assert "lightblue" in dot

    def test_highlight_cleared_on_next_add(self):
        g = CircuitGraph()
        g.add_component("splitter", "1x2", stage=0, lane=0)
        g.add_component("mzm", "2x2", stage=1, lane=0)
        dot = g.to_dot(highlight=True)
        assert "lightblue" in dot
        # Only C2 should be highlighted, not C1
        lines = dot.split("\n")
        c1_line = [l for l in lines if l.strip().startswith("C1 ")][0]
        assert "lightblue" not in c1_line

    def test_rankdir_lr(self):
        g = CircuitGraph()
        g.add_component("splitter", "1x2", stage=0, lane=0)
        dot = g.to_dot()
        assert "rankdir=LR" in dot


# ===== replicate_stage =====================================================

class TestReplicateStage:
    def test_one_to_one_tree(self):
        g = CircuitGraph()
        g.add_component("splitter", "1x2", stage=0, lane=0, role="splitter",
                         description="Root splitter")
        res = g.replicate_stage(
            source_components=["C1"],
            count=2,
            connect_from=["C1.o2", "C1.o3"],
            connection_rule="one_to_one",
            start_stage=1,
        )
        assert res["status"] == "ok"
        assert len(res["created"]) == 2
        assert res["connections_made"] == 2
        # 2 new splitters, each with 2 open output ports + 1 open input
        # Each new 1x2 splitter: o1 (input, connected), o2 (output, open), o3 (output, open)
        # 2 copies × 2 open outputs = 4
        assert len(res["open_ports"]) == 4

    def test_chain_rule(self):
        g = CircuitGraph()
        g.add_component("ring_resonator", "2x2", stage=0, lane=0,
                         role="filter", description="Ring filter")
        res = g.replicate_stage(
            source_components=["C1"],
            count=3,
            connect_from=["C1.o3"],
            connection_rule="chain",
            start_stage=1,
        )
        assert res["status"] == "ok"
        assert len(res["created"]) == 3
        # Chain: C1.o3 -> C2.o1, C2.o3 -> C3.o1, C3.o3 -> C4.o1
        assert res["connections_made"] == 3

    def test_invalid_source(self):
        g = CircuitGraph()
        res = g.replicate_stage(
            source_components=["C99"],
            count=1,
            connect_from=[],
            connection_rule="one_to_one",
        )
        assert res["status"] == "error"

    def test_invalid_connect_from_port(self):
        g = CircuitGraph()
        g.add_component("splitter", "1x2", stage=0, lane=0)
        res = g.replicate_stage(
            source_components=["C1"],
            count=1,
            connect_from=["C1.o99"],
            connection_rule="one_to_one",
        )
        assert res["status"] == "error"
        assert "does not exist" in res["message"]

    def test_already_connected_port(self):
        g = CircuitGraph()
        g.add_component("splitter", "1x2", stage=0, lane=0)
        g.add_component("mzm", "2x2", stage=1, lane=0)
        g.connect("C1", "o2", "C2", "o1")
        res = g.replicate_stage(
            source_components=["C1"],
            count=1,
            connect_from=["C1.o2"],
            connection_rule="one_to_one",
        )
        assert res["status"] == "error"
        assert "already connected" in res["message"]


# ===== PDK Validation ======================================================

_MOCK_CATALOG = [
    PDKEntry(module_name="_mmi1x2", name="mmi1x2", port_config="1x2",
             labels=["passive"], aliases=["splitter", "power splitter"]),
    PDKEntry(module_name="mzi_2x2_pn_diode", name="MZI 2x2 PN Diode",
             port_config="2x2", labels=["modulator", "active"],
             aliases=["amplitude modulator", "pn junction"]),
    PDKEntry(module_name="_gc", name="_gc", port_config="1x0",
             labels=["passive"], aliases=["grating coupler"]),
]


class TestPDKValidation:
    def test_exact_module_name(self):
        g = CircuitGraph(pdk_catalog=_MOCK_CATALOG)
        res = g.add_component("_mmi1x2", "1x2", stage=0, lane=0)
        assert res["status"] == "ok"
        assert res["component_type"] == "_mmi1x2"
        assert res["pdk_name"] == "mmi1x2"

    def test_alias_resolution(self):
        g = CircuitGraph(pdk_catalog=_MOCK_CATALOG)
        res = g.add_component("splitter", "1x2", stage=0, lane=0)
        assert res["status"] == "ok"
        assert res["component_type"] == "_mmi1x2"

    def test_case_insensitive(self):
        g = CircuitGraph(pdk_catalog=_MOCK_CATALOG)
        res = g.add_component("MZI_2X2_PN_DIODE", "2x2", stage=0, lane=0)
        assert res["status"] == "ok"
        assert res["component_type"] == "mzi_2x2_pn_diode"

    def test_wrong_port_config(self):
        g = CircuitGraph(pdk_catalog=_MOCK_CATALOG)
        res = g.add_component("_mmi1x2", "2x2", stage=0, lane=0)
        assert res["status"] == "error"
        assert "1x2" in res["message"]

    def test_unknown_component(self):
        g = CircuitGraph(pdk_catalog=_MOCK_CATALOG)
        res = g.add_component("nonexistent_widget", "2x2", stage=0, lane=0)
        assert res["status"] == "error"
        assert "not found" in res["message"]

    def test_fuzzy_suggestion(self):
        g = CircuitGraph(pdk_catalog=_MOCK_CATALOG)
        res = g.add_component("mzi_2x2", "2x2", stage=0, lane=0)
        assert res["status"] == "error"
        assert "mzi_2x2_pn_diode" in res["message"]

    def test_no_catalog_skips_validation(self):
        g = CircuitGraph()
        res = g.add_component("anything_goes", "2x2", stage=0, lane=0)
        assert res["status"] == "ok"

    def test_replicate_inherits_validation(self):
        g = CircuitGraph(pdk_catalog=_MOCK_CATALOG)
        g.add_component("_mmi1x2", "1x2", stage=0, lane=0)
        res = g.replicate_stage(
            source_components=["C1"], count=2,
            connect_from=["C1.o2"], connection_rule="one_to_one",
        )
        assert res["status"] == "ok"
        assert len(res["created"]) == 2

    def test_dict_catalog_input(self):
        raw_catalog = [
            {"module_name": "_mmi1x2", "name": "mmi1x2", "ports": "1x2",
             "labels": ["passive"], "aka": "splitter, power splitter"},
        ]
        g = CircuitGraph(pdk_catalog=raw_catalog)
        res = g.add_component("splitter", "1x2", stage=0, lane=0)
        assert res["status"] == "ok"
        assert res["component_type"] == "_mmi1x2"


# ===== Detailed DOT Output =================================================

class TestDetailedDot:
    def test_dot_is_digraph(self):
        g = CircuitGraph()
        g.add_component("mzm", "2x2", stage=0, lane=0)
        dot = g.to_dot()
        assert dot.startswith("digraph G")

    def test_dot_shows_ports(self):
        g = CircuitGraph()
        g.add_component("mzm", "2x2", stage=0, lane=0)
        g.add_component("det", "1x0", stage=1, lane=0)
        g.connect("C1", "o3", "C2", "o1")
        dot = g.to_dot()
        assert "o3" in dot
        assert "o1" in dot
        assert "->" in dot

    def test_dot_with_pdk_names(self):
        g = CircuitGraph(pdk_catalog=_MOCK_CATALOG)
        g.add_component("_mmi1x2", "1x2", stage=0, lane=0)
        dot = g.to_dot()
        assert "mmi1x2" in dot

    def test_dot_highlight_new_nodes(self):
        g = CircuitGraph()
        g.add_component("mzm", "2x2", stage=0, lane=0)
        dot = g.to_dot(highlight=True)
        assert "lightblue" in dot

    def test_dot_highlight_new_edges(self):
        g = CircuitGraph()
        g.add_component("mzm", "2x2", stage=0, lane=0)
        g.add_component("det", "2x2", stage=1, lane=0)
        g.connect("C1", "o3", "C2", "o1")
        dot = g.to_dot(highlight=True)
        assert "steelblue" in dot
