"""
Layout & Simulation module — GDS layout generation and SAX circuit simulation.

Not an MCP server. Called directly by the pipeline orchestrator after the user
approves the schematic in the review gate.

Functions:
    render_gds_layout   — instantiate GDSFactory Component, plot layout, build SAX circuit
    run_sax_simulation  — wavelength sweep via SAX, plot S-parameters
    write_gds_file      — write Component to a .gds file with error-handling cascade
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

    try:
        c = _gf.read.from_yaml(netlist_str)
    except Exception:
        data_clean = _clean_netlist_data(dict(data), ignore_links=True)
        netlist_str = yaml.dump(data_clean, default_flow_style=False, sort_keys=False)
        c = _gf.read.from_yaml(netlist_str)
        routing_ok = False

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

    return {
        "gds_fig_b64": gds_b64,
        "required_models": required_models,
        "missing_models": missing,
        "routing_ok": routing_ok,
    }


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
