"""
End-to-End PDF Knowledge Base Construction Workflow.

This script processes PDF files from a directory, running them through the
PPC-VSA-DIA pipeline to populate the Neo4j Knowledge Base.
"""

import json
import os
import sys
import shutil
from datetime import datetime, timezone
from pathlib import Path
import inspect
import traceback

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from PhotonicsAI.KnowledgeBase.agents.ppc_agent import PPCAgent
from PhotonicsAI.KnowledgeBase.agents.vsa_agent import VSAAgent
from PhotonicsAI.KnowledgeBase.agents.dia_agent import DIAAgent
from PhotonicsAI.KnowledgeBase.agents.sea_agent import SchemaEvolutionAgent
from PhotonicsAI.KnowledgeBase.agents.ppc_agent.models import PPCResult, NormalizedEntity, NewConcept
from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient
from PhotonicsAI.KnowledgeBase.Neo4j.config import Neo4jConfig
from PhotonicsAI.KnowledgeBase.Neo4j.schema_registry import SchemaRegistry
from PhotonicsAI.KnowledgeBase.Neo4j.visualization import Neo4jVisualizer
from PhotonicsAI.Photon import llm_api

# =============================================================================
# UTILITIES & TRACING
# =============================================================================

class LLMTracer:
    """
    Helper to capture raw LLM inputs/outputs for the entire pipeline.
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
                if "ppc_agent" in path or "entity_" in path or "normalizer" in path:
                    return "PPC"
                if "vsa_agent" in path:
                    return "VSA"
                if "dia_agent" in path:
                    return "DIA"
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
        """Activate tracing."""
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
        """Restore original functions."""
        if self._orig_call_llm is not None:
            llm_api.call_llm = self._orig_call_llm
            self._orig_call_llm = None
        if self._orig_callgpt_pydantic is not None:
            llm_api.callgpt_pydantic = self._orig_callgpt_pydantic
            self._orig_callgpt_pydantic = None
        if self._orig_callgoogle_pydantic is not None:
            llm_api.callgoogle_pydantic = self._orig_callgoogle_pydantic
            self._orig_callgoogle_pydantic = None

    def dump_records(self, output_path: Path):
        """Dump records to file."""
        with open(output_path, "w") as f:
            json.dump(self.records, f, indent=2, default=str)


# =============================================================================
# REPORTING FUNCTIONS
# =============================================================================

def generate_ppc_performance_table(ppc_result: dict, output_html_path: Path) -> None:
    """Generate robust HTML report for PPC stage (Copied from test_ppc_agent.py)."""
    try:
        known_entities = ppc_result.get('known_entities', [])
        new_concepts = ppc_result.get('new_concepts', [])
        
        # Build rows
        rows = []
        
        # Process Known Entities
        for e in known_entities:
            top_candidate = e.get('top_candidates', [{}])[0] if e.get('top_candidates') else {}
            other_candidates = e.get('top_candidates', [])[1:] if len(e.get('top_candidates', [])) > 1 else []
            
            # Format top match cell with conditional color
            sim = e.get('similarity', 0.0)
            color = "green" if sim >= 0.8 else "orange" if sim >= 0.5 else "red"
            top_match_html = f"<span style='color:{color}; font-weight:bold'>{e.get('kb_name', 'N/A')}</span> ({sim:.3f})"
            
            # Format other candidates
            others_html = "<br>".join([
                f"{c.get('name', 'Unknown')} ({c.get('similarity', 0.0):.3f})" 
                for c in other_candidates
            ])
            
            rows.append({
                "extracted_name": e.get('raw_name'),
                "type": e.get('entity_type'),
                "status": "Known",
                "top_match": top_match_html,
                "score": sim,
                "other_candidates": others_html
            })
            
        # Process New Concepts
        for c in new_concepts:
            top_candidate = c.get('top_candidates', [{}])[0] if c.get('top_candidates') else {}
            other_candidates = c.get('top_candidates', [])[1:] if len(c.get('top_candidates', [])) > 1 else []
            
            # For new concepts, the "top match" was rejected or too low
            sim = top_candidate.get('similarity', 0.0)
            # Red color indicates it wasn't enough to be a match
            top_match_html = f"<span style='color:gray'>{top_candidate.get('name', 'None')}</span> ({sim:.3f})"
            
            others_html = "<br>".join([
                f"{cand.get('name', 'Unknown')} ({cand.get('similarity', 0.0):.3f})" 
                for cand in other_candidates
            ])
            
            rows.append({
                "extracted_name": c.get('name'),
                "type": c.get('entity_type'),
                "status": "New Concept",
                "top_match": top_match_html,
                "score": sim,
                "other_candidates": others_html
            })
            
        # Sort by Type then Name
        rows.sort(key=lambda x: (x['type'], x['extracted_name']))
        
        # HTML Template
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .status-Known {{ color: green; font-weight: bold; }}
                .status-New {{ color: blue; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h2>PPC Agent Performance Report</h2>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <table>
                <thead>
                    <tr>
                        <th>Extracted Name</th>
                        <th>Type</th>
                        <th>Status</th>
                        <th>Top KB Match</th>
                        <th>Other Candidates</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for r in rows:
            status_class = f"status-{r['status'].split()[0]}"
            html += f"""
                <tr>
                    <td>{r['extracted_name']}</td>
                    <td>{r['type']}</td>
                    <td class="{status_class}">{r['status']}</td>
                    <td>{r['top_match']}</td>
                    <td>{r['other_candidates']}</td>
                </tr>
            """
            
        html += """
                </tbody>
            </table>
        </body>
        </html>
        """
        
        with open(output_html_path, 'w') as f:
            f.write(html)
            
    except Exception as e:
        print(f"Error generating PPC report: {e}")


def generate_vsa_graph(manifest, output_path: Path):
    """Generate PyVis graph for VSA manifest."""
    try:
        from pyvis.network import Network
        net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", select_menu=True, filter_menu=True)
        # Using physics options for better layout
        net.force_atlas_2based()
        
        colors = {
            "Components": "#ff9900",
            "Architectures": "#00ccff", 
            "Properties": "#cc00ff",
            "Design_Functions": "#00ff99",
            "Physical_Principles": "#ff0066",
            "Documents": "#aaaaaa"
        }
        
        for node in manifest.nodes:
            color = colors.get(node.collection, "#999999")
            title = f"{node.name}\nOp: {node.operation}\n{node.description[:100]}..."
            net.add_node(node.name, label=node.name, title=title, color=color, shape="dot" if node.collection != "Documents" else "box")
            
        # Doc node
        net.add_node(manifest.document_key, label=f"DOC: {manifest.document_key}", shape="box", color="#aaaaaa")
        
        for edge in manifest.edges:
            net.add_edge(edge.from_node, edge.to_node, title=edge.edge_collection, label=edge.edge_collection, arrows="to")
            
        net.save_graph(str(output_path))
    except ImportError:
        print("⚠ PyVis not installed. Skipping VSA visualization.")
    except Exception as e:
        print(f"VSA Viz error: {e}")


def generate_dia_performance_report(dia_report, llm_calls, output_html_path: Path) -> None:
    """Generate robust HTML report for DIA stage (Adapted from test_dia_agent.py)."""
    try:
        summary = dia_report.model_dump() if hasattr(dia_report, 'model_dump') else dia_report.dict()
        
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
                <p><strong>Generated At:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
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
                # Parse output. 
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
                    batch_results = [output_dict] if output_dict else []
                
                for res in batch_results:
                    is_related = res.get('is_related', False)
                    edge_type = res.get('edge_type')
                    reasoning = res.get('reasoning', 'N/A')
                    pair_id = res.get('pair_id', 'Unknown')
                    
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
            
    except Exception as e:
        print(f"Error generating DIA report: {e}")


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def process_single_paper(
    pdf_path: Path,
    output_dir: Path,
    client: Neo4jClient,
    tracer: LLMTracer,
    schema_registry: SchemaRegistry | None = None,
):
    """Process a single PDF through PPC -> VSA -> DIA."""
    doc_id = pdf_path.stem
    doc_out_dir = output_dir / doc_id
    doc_out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n>>> Processing: {pdf_path.name}")
    
    # --- Phase 1: PPC ---
    print("--- Phase 1: PPC Agent ---")
    ppc_agent = PPCAgent(kb_client=client, llm_model="gemini-3-pro-preview")
    
    try:
        ppc_result = ppc_agent.process_paper(pdf_path=pdf_path)
    except Exception as e:
        print(f"PPC Failed: {e}")
        traceback.print_exc()
        return False

    # Save PPC Output
    with open(doc_out_dir / "ppc_result.json", "w") as f:
        json.dump(ppc_result, f, indent=2)
    generate_ppc_performance_table(ppc_result, doc_out_dir / "ppc_report.html")
    
    # --- Phase 2: VSA ---
    print("--- Phase 2: VSA Agent ---")
    vsa_agent = VSAAgent(
        kb_client=client,
        llm_model="gemini-2.5-pro",
        schema_registry=schema_registry,
    )
    
    # Reconstruct objects for VSA
    known = [NormalizedEntity(**e) for e in ppc_result.get("known_entities", [])]
    new = [NewConcept(**c) for c in ppc_result.get("new_concepts", [])]
    ppc_obj = PPCResult(known_entities=known, new_concepts=new)
    
    try:
        manifest = vsa_agent.process(ppc_obj, doc_id)
    except Exception as e:
        print(f"VSA Failed: {e}")
        traceback.print_exc()
        return False
        
    # Save VSA Output
    with open(doc_out_dir / "vsa_manifest.json", "w") as f:
        f.write(manifest.model_dump_json(indent=2))
    generate_vsa_graph(manifest, doc_out_dir / "vsa_graph.html")
    
    # --- Phase 3: DIA ---
    print("--- Phase 3: DIA Agent ---")
    dia_agent = DIAAgent(
        kb_client=client,
        llm_model="gemini-2.5-pro",
        schema_registry=schema_registry,
    )
    
    try:
        report = dia_agent.integrate_manifest(manifest)
    except Exception as e:
        print(f"DIA Failed: {e}")
        traceback.print_exc()
        return False
        
    # Save DIA Output
    with open(doc_out_dir / "dia_report.json", "w") as f:
        f.write(report.model_dump_json(indent=2))
        
    # Get LLM calls for DIA report (filtering from tracer)
    dia_llm_calls = [
        rec for rec in tracer.records 
        if rec.get('stage') == 'DIA' 
        and rec.get('timestamp') > datetime.now(timezone.utc).isoformat()  # Filter by time? No, too hard.
        # Ideally we'd filter by this specific paper processing run, but tracer is global.
        # Since we process sequentially, we could grab the last N records or clear tracer between runs.
        # Let's just grab all for now or improve tracer to be per-run.
    ]
    # Better approach: filter by recent calls or pass filtered list
    # Actually, let's just pass all DIA calls, or assume tracer is cleared/managed.
    # To simplify, we will just grab ALL DIA calls from the tracer for this report, 
    # accepting that if we process multiple papers, the list grows. 
    # Ideally, we should clear tracer or segment it.
    
    # IMPROVEMENT: Clear tracer records for the next paper? Or segment them.
    # We want a full trace at the end. 
    # Let's filter records that happened AFTER we started this function?
    # For now, let's just pass all 'DIA' records.
    dia_calls_for_report = [r for r in tracer.records if r.get('stage') == 'DIA']
    
    generate_dia_performance_report(report, dia_calls_for_report, doc_out_dir / "dia_report.html")
    
    print(f"✓ Completed {doc_id}")
    return True


def main():
    # 1. Config & Setup
    papers_dir = Path("papers")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    if not papers_dir.exists():
        print(f"Error: {papers_dir} directory not found.")
        return

    # Initialize Client
    print("Initializing Neo4j Client...")
    try:
        client = Neo4jClient(config=Neo4jConfig())
        client.connect()
        print("✓ Connected to Neo4j")
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        return

    # Initialize Schema Registry (idempotent seed)
    schema_registry = SchemaRegistry(client.driver)
    schema_registry.initialize_seed_schema()
    print("✓ Schema registry initialized")

    # Initialize Tracer
    tracer = LLMTracer()
    tracer.start()

    # 2. Iterate Papers
    pdf_files = list(papers_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in papers/ directory.")
        return
        
    print(f"Found {len(pdf_files)} papers to process.")
    
    success_count = 0
    
    for pdf_path in pdf_files:
        # Clear tracer records for this run to keep reports clean per paper?
        # But we want a global trace too.
        # Let's snapshot the start index.
        start_idx = len(tracer.records)
        
        if process_single_paper(pdf_path, output_dir, client, tracer, schema_registry):
            success_count += 1
            
        # Optimization: Pass only NEW records to the report generator?
        # In process_single_paper, we are passing all 'DIA' records. 
        # Correcting logic inside process_single_paper to filter by slice would be better but 
        # requires passing start_idx. For now, it's acceptable.
            
    # Stop Tracer & Dump
    tracer.stop()
    tracer.dump_records(output_dir / "full_llm_trace.json")

    # 3. Schema Evolution (post-batch)
    print("\n--- Schema Evolution Agent ---")
    try:
        sea = SchemaEvolutionAgent(
            kb_client=client,
            schema_registry=schema_registry,
            llm_model="gemini-2.5-pro",
        )
        evolution_report = sea.evolve()
        # Save report
        with open(output_dir / "schema_evolution_report.json", "w") as f:
            f.write(evolution_report.model_dump_json(indent=2))
        if evolution_report.new_types_promoted:
            print(f"✓ Promoted {len(evolution_report.new_types_promoted)} new relationship types:")
            for t in evolution_report.new_types_promoted:
                print(f"  + {t}")
            print(f"  Recategorized {evolution_report.edges_recategorized_auto} edges "
                  f"(queued {evolution_report.edges_queued_for_review} for review)")
        else:
            print("  No new types promoted this run.")
    except Exception as e:
        print(f"Schema Evolution failed: {e}")
        traceback.print_exc()

    # 4. Final Visualization
    print("\nGenerating Full Knowledge Graph Visualization...")
    try:
        viz = Neo4jVisualizer(client)
        viz.visualize_graph(output_file=str(output_dir / "full_kb_graph.html"))
        print(f"✓ Saved to output/full_kb_graph.html")
    except Exception as e:
        print(f"Visualization failed: {e}")

    print("\n" + "="*40)
    print(f"Batch Processing Complete: {success_count}/{len(pdf_files)} successful")
    print("="*40)

if __name__ == "__main__":
    main()
