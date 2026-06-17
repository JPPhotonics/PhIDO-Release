"""
Layout & Simulation module — GDS layout generation and SAX circuit simulation.

Not an MCP server. Called directly by the pipeline orchestrator after the user
approves the schematic in the review gate.

Functions:
    render_gds_layout   — instantiate GDSFactory Component, plot layout, build SAX circuit
    run_sax_simulation  — wavelength sweep via SAX, plot S-parameters
    write_gds_file      — write Component to a .gds file with error-handling cascade
    run_drc_check       — run KLayout DRC on a .gds and return a structured pass/fail
"""

import base64
import io
import json
import uuid
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).parent.parent
BUILD_DIR = REPO_ROOT / "build"
BUILD_DIR.mkdir(parents=True, exist_ok=True)
PDK_DIR = REPO_ROOT / "PhotonicsAI" / "KnowledgeBase" / "DesignLibrary"

# ---------------------------------------------------------------------------
# Lazy-loaded globals
# ---------------------------------------------------------------------------

_gf = None
_pdk = None
_all_models: Optional[dict] = None
_cached_component = None          # last gf.Component built
_cached_sax_circuit = None        # last SAX circuit callable
_cached_netlist_hash: Optional[int] = None  # hash of the netlist YAML used

def _list_pdk_modules() -> list[str]:
    return [f.stem for f in PDK_DIR.glob("*.py") if f.name != "__init__.py"]


def _ensure_pdk():
    """Activate GDSFactory + DemoPDK on first call."""
    global _gf, _pdk
    if _gf is not None:
        return

    import importlib
    import gdsfactory as gf_lib
    from gdsfactory.generic_tech import get_generic_pdk

    generic = get_generic_pdk()
    cells = {}
    for mod_name in _list_pdk_modules():
        full = f"PhotonicsAI.KnowledgeBase.DesignLibrary.{mod_name}"
        module = importlib.import_module(full)
        cells[mod_name] = getattr(module, mod_name)

    pdk = gf_lib.Pdk(
        name="DemoPDK",
        layers=generic.layers,
        cross_sections=generic.cross_sections,
        cells=cells,
        layer_views=generic.layer_views,
    )
    pdk.activate()

    _gf = gf_lib
    _pdk = pdk


def _ensure_models():
    """Load SAX models from every DesignLibrary module on first call."""
    global _all_models
    if _all_models is not None:
        return

    import importlib
    models: dict = {}
    for mod_name in _list_pdk_modules():
        full = f"PhotonicsAI.KnowledgeBase.DesignLibrary.{mod_name}"
        try:
            module = importlib.import_module(full)
            models.update(module.get_model())
        except Exception as exc:
            print(f"Warning: could not load SAX model from {mod_name}: {exc}")
    _all_models = models


def _fig_to_b64(fig) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _clean_netlist_data(data: dict, ignore_links: bool = False) -> dict:
    """Sanitise a GDSFactory netlist dict before instantiation."""
    if ignore_links and "routes" in data:
        del data["routes"]
    if "name" in data:
        data["name"] = f"{data['name']}_{uuid.uuid4().hex[:8]}"
    for key in ("reasoning", "comments"):
        data.pop(key, None)
    if "placements" not in data:
        data["placements"] = {}
        x = y = 0
        for instance in data.get("instances", {}):
            data["placements"][instance] = {"x": x, "y": y}
            x += 10
            y += 10
    return data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_gds_layout(gf_netlist_yaml: str) -> dict:
    """Instantiate the GDSFactory Component, build the SAX circuit, and render the layout.

    Returns a dict with:
        gds_fig_b64       — base64-encoded PNG of the layout plot
        required_models   — list of SAX model names needed
        missing_models    — subset of required_models not in the PDK
        routing_ok        — True if optical routing succeeded
    """
    global _cached_component, _cached_sax_circuit, _cached_netlist_hash

    _ensure_pdk()
    _ensure_models()
    import sax
    import matplotlib.pyplot as plt

    data = yaml.safe_load(gf_netlist_yaml)
    routing_ok = True

    data_clean = _clean_netlist_data(dict(data), ignore_links=False)
    netlist_str = yaml.dump(data_clean, default_flow_style=False, sort_keys=False)

    routing_warnings: list[str] = []
    routing_error_detail = ""

    links_count = 0
    for _rname, rdata in data_clean.get("routes", {}).items():
        links_count += len(rdata.get("links", {}))

    try:
        c = _gf.read.from_yaml(netlist_str)
    except RuntimeError as collision_exc:
        if "collision" in str(collision_exc).lower():
            routing_warnings.append(
                f"Routing collision detected: {collision_exc}. "
                "Routes are drawn but may overlap component bounding boxes. "
                "This is cosmetic and does not affect simulation accuracy."
            )
            for route_bundle in data_clean.get("routes", {}).values():
                if isinstance(route_bundle, dict) and "settings" in route_bundle:
                    route_bundle["settings"]["on_collision"] = None
            netlist_str = yaml.dump(data_clean, default_flow_style=False, sort_keys=False)
            c = _gf.read.from_yaml(netlist_str)
            routing_ok = True
        else:
            raise
    except Exception as routing_exc:
        import traceback
        routing_error_detail = str(routing_exc)
        traceback.print_exc()
        routing_warnings.append(
            f"Routing failed ({type(routing_exc).__name__}): {routing_exc}. "
            "Layout rendered without optical routes."
        )
        data_clean = _clean_netlist_data(dict(data), ignore_links=True)
        netlist_str = yaml.dump(data_clean, default_flow_style=False, sort_keys=False)
        try:
            c = _gf.read.from_yaml(netlist_str)
        except Exception:
            traceback.print_exc()
            raise
        routing_ok = False

    try:
        flat_netlist = c.get_netlist(recursive=False)
        conn_count = len(flat_netlist.get("connections", {}))
        route_count = len(flat_netlist.get("routes", {}))
        if links_count > 0 and conn_count == 0 and route_count == 0 and routing_ok:
            routing_warnings.append(
                f"Expected {links_count} routed connections but "
                "the generated component has none."
            )
    except Exception:
        pass

    try:
        recursive_netlist = c.get_netlist(recursive=True)
        required_models = sax.get_required_circuit_models(recursive_netlist)
    except Exception:
        recursive_netlist = c.get_netlist(recursive=False)
        required_models = sax.get_required_circuit_models(recursive_netlist)

    missing = [m for m in required_models if m not in _all_models]

    try:
        _circuit, _info = sax.circuit(recursive_netlist, _all_models, backend="default")
    except Exception as exc:
        _circuit = None
        missing.append(f"SAX_BUILD_ERROR: {exc}")

    gds_fig = c.plot(return_fig=True, show_labels=True)
    gds_fig.savefig(str(BUILD_DIR / "plot_gds.png"), dpi=150, bbox_inches="tight")
    gds_b64 = _fig_to_b64(gds_fig)

    _cached_component = c
    _cached_sax_circuit = _circuit
    _cached_netlist_hash = hash(gf_netlist_yaml)

    result = {
        "gds_fig_b64": gds_b64,
        "required_models": required_models,
        "missing_models": missing,
        "routing_ok": routing_ok,
        "routing_warnings": routing_warnings,
    }
    if routing_error_detail:
        result["routing_error"] = routing_error_detail
    return result


def run_sax_simulation(
    gf_netlist_yaml: str,
    wl_start: float = 1.5,
    wl_stop: float = 1.6,
    wl_points: int = 200,
) -> dict:
    """Run a wavelength-sweep SAX simulation and plot S-parameters.

    If ``render_gds_layout`` was called with the same netlist, reuses the
    cached SAX circuit. Otherwise builds it from scratch.

    Returns a dict with:
        sax_fig_b64  — base64-encoded PNG of the S-parameter plot
        s_params     — dict mapping "port_in,port_out" to list of |S|^2 in dB
        wavelengths  — list of wavelength values
        error        — error string if simulation failed, else null
    """
    global _cached_sax_circuit, _cached_netlist_hash

    _ensure_pdk()
    _ensure_models()
    import numpy as np
    import matplotlib.pyplot as plt
    import sax

    circuit_fn = _cached_sax_circuit
    if circuit_fn is None or hash(gf_netlist_yaml) != _cached_netlist_hash:
        data = yaml.safe_load(gf_netlist_yaml)
        data = _clean_netlist_data(dict(data), ignore_links=False)
        netlist_str = yaml.dump(data, default_flow_style=False, sort_keys=False)
        try:
            c = _gf.read.from_yaml(netlist_str)
            recursive = c.get_netlist(recursive=True)
            circuit_fn, _ = sax.circuit(recursive, _all_models, backend="default")
        except Exception as exc:
            return {"sax_fig_b64": "", "s_params": {}, "wavelengths": [], "error": str(exc)}

    wl = np.linspace(wl_start, wl_stop, wl_points)
    try:
        result = circuit_fn(wl=wl)
    except Exception as exc:
        return {"sax_fig_b64": "", "s_params": {}, "wavelengths": [], "error": str(exc)}

    filtered = {k: v for k, v in result.items() if k[0] == "o1"}

    s_params_json: dict[str, list[float]] = {}
    for (p_in, p_out), arr in filtered.items():
        key = f"{p_in},{p_out}"
        mag_db = (10 * np.log10(np.abs(np.asarray(arr)) ** 2)).tolist()
        s_params_json[key] = mag_db

    cols = min(6, max(1, len(filtered)))
    num_plots = len(filtered)
    rows = max(1, (num_plots + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2 * rows))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for ax, (key, arr) in zip(axes, filtered.items()):
        mag = 10 * np.log10(np.abs(np.asarray(arr)) ** 2)
        ax.plot(wl, mag, linewidth=2, color="darksalmon")
        ax.text(
            0.95, 0.95, f"{key[0]}-{key[1]}",
            ha="right", va="top", transform=ax.transAxes,
        )

    for ax in axes[num_plots:]:
        fig.delaxes(ax)

    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(wspace=0.2, hspace=0.1)
    fig.savefig(str(BUILD_DIR / "plot_sax.png"), dpi=150, bbox_inches="tight")
    sax_b64 = _fig_to_b64(fig)

    return {
        "sax_fig_b64": sax_b64,
        "s_params": s_params_json,
        "wavelengths": wl.tolist(),
        "error": None,
    }


def write_gds_file(
    gf_netlist_yaml: str,
    filename: str = "circuit",
) -> dict:
    """Write the GDSFactory Component to a .gds file.

    Reuses the cached Component from ``render_gds_layout`` when possible.

    Returns ``{"gds_path": str, "success": bool, "error": str|None}``.
    """
    _ensure_pdk()

    c = _cached_component
    if c is None or hash(gf_netlist_yaml) != _cached_netlist_hash:
        data = yaml.safe_load(gf_netlist_yaml)
        data = _clean_netlist_data(dict(data), ignore_links=False)
        netlist_str = yaml.dump(data, default_flow_style=False, sort_keys=False)
        try:
            c = _gf.read.from_yaml(netlist_str)
        except Exception as exc:
            return {"gds_path": "", "success": False, "error": str(exc)}

    gds_path = str(BUILD_DIR / f"{filename}.gds")

    try:
        c.write_gds(gds_path)
        return {"gds_path": gds_path, "success": True, "error": None}
    except Exception as exc:
        if "layer numbers larger than 65535" in str(exc):
            try:
                flattened = c.flatten()
                if flattened is not None:
                    flattened.write_gds(gds_path)
                    return {"gds_path": gds_path, "success": True, "error": None}
            except Exception:
                pass
            try:
                c.write_gds(gds_path, max_points=None)
                return {"gds_path": gds_path, "success": True, "error": None}
            except Exception as final_exc:
                return {"gds_path": "", "success": False, "error": str(final_exc)}
        return {"gds_path": "", "success": False, "error": str(exc)}


def run_drc_check(gds_path: str, timeout: int = 120) -> dict:
    """Run KLayout DRC on a GDS file and return a structured pass/fail result.

    Wraps the same batch-mode KLayout DRC the baseline pipeline uses
    (``PhotonicsAI/Photon/drc/drc_script.drc``) and parses the resulting
    report database (.lydrb, an XML report-database with one ``<item>`` per
    violation) so the agentic pipeline can emit a machine-readable signal.

    Args:
        gds_path: Path to the .gds file to check (from ``write_gds_file``).
        timeout: KLayout subprocess timeout in seconds.

    Returns a dict with:
        drc_ran      — True if KLayout executed the DRC script
        drc_clean    — True if zero violations, False if >0, None if undetermined
        violations   — number of violation items (-1 if undetermined)
        report_path  — path to the .lydrb report (empty if not produced)
        error        — error string if DRC could not run, else None
    """
    import shutil
    import subprocess
    import xml.etree.ElementTree as ET

    def _fail(error: str, ran: bool = False, report: str = "") -> dict:
        return {"drc_ran": ran, "drc_clean": None, "violations": -1,
                "report_path": report, "error": error}

    drc_script = REPO_ROOT / "PhotonicsAI" / "Photon" / "drc" / "drc_script.drc"
    report_path = BUILD_DIR / "report.lydrb"

    if not gds_path or not Path(gds_path).exists():
        return _fail(f"GDS file not found: {gds_path}")
    if not drc_script.exists():
        return _fail(f"DRC script not found: {drc_script}")

    klayout = shutil.which("klayout") or next(
        (p for p in ("/usr/bin/klayout", "/usr/local/bin/klayout",
                     "/opt/klayout/bin/klayout") if Path(p).exists()),
        None,
    )
    if klayout is None:
        return _fail("KLayout executable not found on PATH")

    try:
        proc = subprocess.run(
            [klayout, "-b", "-r", str(drc_script),
             "-rd", f"input_gds={gds_path}", "-rd", f"report={report_path}"],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return _fail(f"DRC timed out after {timeout}s")
    except Exception as exc:
        return _fail(f"DRC subprocess failed: {exc}")

    if proc.returncode != 0:
        return _fail(f"KLayout returned {proc.returncode}: {proc.stderr.strip()[:500]}")
    if not report_path.exists():
        return _fail("DRC ran but no report database was produced", ran=True)

    try:
        tree = ET.parse(report_path)
        violations = sum(1 for _ in tree.getroot().iter("item"))
    except ET.ParseError as exc:
        return {"drc_ran": True, "drc_clean": None, "violations": -1,
                "report_path": str(report_path),
                "error": f"Could not parse DRC report: {exc}"}

    return {
        "drc_ran": True,
        "drc_clean": violations == 0,
        "violations": violations,
        "report_path": str(report_path),
        "error": None,
    }
