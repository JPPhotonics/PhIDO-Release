"""Test script for DIA Agent."""

import json
import os
from datetime import datetime, timezone
import inspect
from pathlib import Path

from PhotonicsAI.KnowledgeBase.agents.dia_agent import DIAAgent
from PhotonicsAI.KnowledgeBase.agents.vsa_agent.models import VSAUpdatePayload
from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient as KnowledgeBaseClient
from PhotonicsAI.KnowledgeBase.Neo4j.config import Neo4jConfig
from PhotonicsAI.Photon import llm_api

class LLMTracer:
    """
    Helper to capture raw LLM inputs/outputs for the DIA agent pipeline.
    """

    def __init__(self) -> None:
        self.records = []
        self._orig_call_llm = None
        self._orig_callgpt_pydantic = None
        self._orig_callgoogle_pydantic = None

    def _detect_stage(self) -> str:
        """Best-effort detection of which phase made the LLM call."""
        try:
            for frame_info in inspect.stack():
                module = inspect.getmodule(frame_info.frame)
                if not module or not getattr(module, "__file__", None):
                    continue
                path = module.__file__
                if "dia_agent" in path:
                    return "semantic_evidence_mining"
        except Exception:
            pass
        return "other"

    def _wrap_call_llm(self, *args, **kwargs):
        stage = self._detect_stage()
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "api": "call_llm",
            "stage": stage,
            "args": repr(args),
            "kwargs": repr(kwargs),
        }
        try:
            result = self._orig_call_llm(*args, **kwargs)
            record["output"] = repr(result)
            record["error"] = None
            return result
        except Exception as e:
            record["output"] = None
            record["error"] = repr(e)
            raise
        finally:
            self.records.append(record)

    def _wrap_callgpt_pydantic(self, *args, **kwargs):
        stage = self._detect_stage()
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "api": "callgpt_pydantic",
            "stage": stage,
            "args": repr(args),
            "kwargs": repr(kwargs),
        }
        try:
            result = self._orig_callgpt_pydantic(*args, **kwargs)
            try:
                record["output"] = result.model_dump()
            except Exception:
                record["output"] = repr(result)
            record["error"] = None
            return result
        except Exception as e:
            record["output"] = None
            record["error"] = repr(e)
            raise
        finally:
            self.records.append(record)

    def _wrap_callgoogle_pydantic(self, *args, **kwargs):
        stage = self._detect_stage()
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "api": "callgoogle_pydantic",
            "stage": stage,
            "args": repr(args),
            "kwargs": repr(kwargs),
        }
        try:
            result = self._orig_callgoogle_pydantic(*args, **kwargs)
            try:
                record["output"] = result.model_dump()
            except Exception:
                record["output"] = repr(result)
            record["error"] = None
            return result
        except Exception as e:
            record["output"] = None
            record["error"] = repr(e)
            raise
        finally:
            self.records.append(record)

    def start(self) -> None:
        """Activate tracing by monkeypatching llm_api."""
        if self._orig_call_llm is None:
            self._orig_call_llm = llm_api.call_llm
            llm_api.call_llm = self._wrap_call_llm
        if self._orig_callgpt_pydantic is None and hasattr(llm_api, "callgpt_pydantic"):
            self._orig_callgpt_pydantic = llm_api.callgpt_pydantic
            llm_api.callgpt_pydantic = self._wrap_callgpt_pydantic
        if self._orig_callgoogle_pydantic is None and hasattr(llm_api, "callgoogle_pydantic"):
            self._orig_callgoogle_pydantic = llm_api.callgoogle_pydantic
            llm_api.callgoogle_pydantic = self._wrap_callgoogle_pydantic

    def stop(self) -> None:
        """Restore original llm_api functions."""
        if self._orig_call_llm is not None:
            llm_api.call_llm = self._orig_call_llm
            self._orig_call_llm = None
        if self._orig_callgpt_pydantic is not None:
            llm_api.callgpt_pydantic = self._orig_callgpt_pydantic
            self._orig_callgpt_pydantic = None
        if self._orig_callgoogle_pydantic is not None:
            llm_api.callgoogle_pydantic = self._orig_callgoogle_pydantic
            self._orig_callgoogle_pydantic = None

    def write_report(self, output_path: Path, dia_report, manifest_stats: dict) -> None:
        """Persist a detailed JSON report for the DIA pipeline."""
        grouped = {
            "semantic_evidence_mining": [],
            "other": [],
        }
        for rec in self.records:
            stage = rec.get("stage", "other")
            key = stage if stage in grouped else "other"
            grouped[key].append(rec)

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "manifest_stats": manifest_stats,
            "integration_summary": dia_report.model_dump() if hasattr(dia_report, 'model_dump') else dia_report.dict(),
            "llm_calls": grouped,
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)


def generate_dia_performance_report(json_path: Path, output_html_path: Path) -> None:
    """
    Generate an HTML performance report for DIA integration.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        summary = data.get('integration_summary', {})
        llm_calls = data.get('llm_calls', {}).get('semantic_evidence_mining', [])
        
        # HTML Template
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: sans-serif; margin: 20px; background-color: #f4f4f9; }}
                .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                h2 {{ color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
                h3 {{ color: #555; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
                th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
                th {{ background-color: #f8f9fa; color: #444; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .metric-card {{ display: inline-block; width: 22%; margin: 1%; padding: 15px; background: #f8f9fa; border-radius: 5px; text-align: center; }}
                .metric-val {{ font-size: 24px; font-weight: bold; color: #007bff; display: block; }}
                .metric-label {{ font-size: 14px; color: #666; }}
                .success {{ color: green; font-weight: bold; }}
                .failure {{ color: red; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>DIA Integration Report</h2>
                <p><strong>Generated At:</strong> {data.get('generated_at')}</p>
                <p><strong>Document Key:</strong> {summary.get('document_key')}</p>
                
                <div class="metrics">
                    <div class="metric-card">
                        <span class="metric-val">{summary.get('nodes_created', 0)}</span>
                        <span class="metric-label">Nodes Created</span>
                    </div>
                    <div class="metric-card">
                        <span class="metric-val">{summary.get('nodes_updated', 0)}</span>
                        <span class="metric-label">Nodes Updated</span>
                    </div>
                    <div class="metric-card">
                        <span class="metric-val">{summary.get('explicit_edges_created', 0)}</span>
                        <span class="metric-label">Explicit Edges</span>
                    </div>
                    <div class="metric-card">
                        <span class="metric-val">{summary.get('inferred_edges_found', 0)}</span>
                        <span class="metric-label">Inferred Edges</span>
                    </div>
                </div>

                <h3>Global Inference (Semantic Evidence Mining)</h3>
                <p>Total LLM Verification Calls: {len(llm_calls)}</p>
                
                <table>
                    <thead>
                        <tr>
                            <th>Candidate Pair</th>
                            <th>Verification Result</th>
                            <th>Reasoning</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for call in llm_calls:
            try:
                # Parse output. It should be a dict or a serialized dict containing 'results' list for batches
                output = call.get('output')
                if isinstance(output, str):
                    try:
                        import ast
                        output_dict = ast.literal_eval(output)
                    except:
                        output_dict = {}
                else:
                    output_dict = output if output else {}
                
                # Check for batch results
                batch_results = output_dict.get('results', [])
                if not batch_results:
                    # Single result fallback (if any legacy calls remain or fallback logic)
                    batch_results = [output_dict] if output_dict else []
                
                for res in batch_results:
                    is_related = res.get('is_related', False)
                    edge_type = res.get('edge_type')
                    reasoning = res.get('reasoning', 'N/A')
                    pair_id = res.get('pair_id', 'Unknown')
                    
                    # Try to split pair_id to get names: "NodeName::CandidateName"
                    if "::" in pair_id:
                        parts = pair_id.split("::")
                        pair_display = f"{parts[0]} <-> {parts[1]}"
                    else:
                        pair_display = pair_id
                    
                    status_class = "success" if is_related else "failure"
                    status_text = f"YES ({edge_type})" if is_related else "NO"
                    
                    html += f"""
                        <tr>
                            <td>{pair_display}</td>
                            <td class="{status_class}">{status_text}</td>
                            <td>{reasoning}</td>
                        </tr>
                    """
            except Exception as e:
                print(f"Error parsing call: {e}")
                continue
                
        html += """
                    </tbody>
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(output_html_path, 'w') as f:
            f.write(html)
            
        print(f"✓ DIA Performance Report saved to {output_html_path}")
        
    except Exception as e:
        print(f"Error generating DIA report: {e}")


def _setup_real_agent():
    """Setup DIA Agent with real Neo4j connection."""
    config = Neo4jConfig()
    
    print(f"Connecting to Neo4j at {config.uri}...")
    try:
        client = KnowledgeBaseClient(config=config)
        client.connect()
        print("✓ Connected to Neo4j")
        return DIAAgent(kb_client=client, llm_model="gemini-2.5-pro")
    except Exception as e:
        print(f"X Failed to connect to Neo4j: {e}")
        print("  Make sure Neo4j is running.")
        raise e

def _visualize_knowledgebase(client, output_filename="dia_knowledgebase.html"):
    """Visualize the entire knowledge base using PyVis."""
    try:
        from pyvis.network import Network
        from PhotonicsAI.KnowledgeBase.Neo4j.visualization import Neo4jVisualizer
        
        print(f"\n[Viz] Generating full KB visualization: {output_filename}...")
        
        # Neo4j Logic
        print("  Using Neo4j visualization logic...")
        viz = Neo4jVisualizer(client)
        viz.visualize_graph(output_file=output_filename)
        print(f"✓ KB Visualization saved to {output_filename}")
        return

    except ImportError:
        print("⚠ PyVis or Visualizer not found. Skipping.")
        return
    except Exception as e:
        print(f"⚠ Visualization failed: {e}")



def test_dia_pipeline():
    """Test full DIA pipeline with VSA manifest."""
    print("\n" + "=" * 60)
    print("Test: DIA Agent Pipeline")
    print("=" * 60)
    
    # Load manifest
    manifest_path = Path("vsa_manifest_real_ppc_output_test.json")
    if not manifest_path.exists():
        print(f"ℹ Manifest {manifest_path} not found. Run VSA test first.")
        return

    print(f"Loading manifest from {manifest_path}...")
    with open(manifest_path, 'r') as f:
        data = json.load(f)
        manifest = VSAUpdatePayload(**data)
        
    print(f"Loaded manifest with {len(manifest.nodes)} nodes and {len(manifest.edges)} edges.")
    
    try:
        agent = _setup_real_agent()
        
        tracer = LLMTracer()
        tracer.start()
        
        try:
            # Run Integration
            print("\n[DIA] Executing Integration...")
            report = agent.integrate_manifest(manifest)
        finally:
            tracer.stop()
        
        print("\n[DIA] Report Summary:")
        print(f"  - Document: {report.document_key}")
        print(f"  - Nodes Created: {report.nodes_created}")
        print(f"  - Nodes Updated: {report.nodes_updated}")
        print(f"  - Nodes Merged: {report.nodes_merged}")
        print(f"  - Explicit Edges: {report.explicit_edges_created}")
        print(f"  - Inferred Edges: {report.inferred_edges_found}")
        print(f"  - Total Edges: {report.total_edges_created}")
        print(f"  - Errors: {len(report.errors)}")
        
        if report.errors:
            print("\n  [Errors]:")
            for err in report.errors:
                print(f"    - {err}")
                
        # Basic assertions
        if report.total_edges_created > 0:
            print("\n✓ Success: Edges were created in the database.")
        else:
            print("\n⚠ Warning: No edges created (possibly already existed or empty manifest).")
            
        # Generate Reports
        report_json_path = Path("dia_agent_detailed_report.json")
        tracer.write_report(
            report_json_path, 
            report, 
            {"nodes": len(manifest.nodes), "edges": len(manifest.edges)}
        )
        print(f"\n✓ Detailed LLM Report saved to {report_json_path}")
        
        report_html_path = Path("dia_performance_report.html")
        generate_dia_performance_report(report_json_path, report_html_path)
            
        # Visualize Result
        _visualize_knowledgebase(agent.kb_client, "dia_knowledgebase_viz.html")

    except Exception as e:
        print(f"\nX Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dia_pipeline()
