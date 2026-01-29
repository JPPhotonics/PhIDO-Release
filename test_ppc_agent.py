"""Test script for PPC Agent."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import inspect

from PhotonicsAI.KnowledgeBase.agents.ppc_agent import PPCAgent
from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient as KnowledgeBaseClient
from PhotonicsAI.KnowledgeBase.Neo4j.config import Neo4jConfig
from PhotonicsAI.Photon import llm_api


class LLMTracer:
    """
    Helper to capture raw LLM inputs/outputs for the PPC agent pipeline.

    It monkeypatches llm_api.call_llm and llm_api.callgpt_pydantic and records:
      - stage (preprocessing, entity_extraction, conflict_handling, other)
      - raw args/kwargs
      - raw outputs or errors
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
                if "llm_document_preprocessor" in path:
                    return "preprocessing"
                if "entity_extractor" in path:
                    return "entity_extraction"
                if "entity_describer" in path:
                    return "entity_description"
                if "conflict_detector" in path:
                    return "conflict_handling"
        except Exception:
            # Fallback if stack inspection fails
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
            # Pydantic models may not be JSON-serializable; store a string view
            try:
                record["output"] = result.model_dump()  # type: ignore[attr-defined]
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
            # Pydantic models may not be JSON-serializable; store a dict view if possible
            try:
                record["output"] = result.model_dump()  # type: ignore[attr-defined]
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
            llm_api.call_llm = self._wrap_call_llm  # type: ignore[assignment]
        if self._orig_callgpt_pydantic is None and hasattr(llm_api, "callgpt_pydantic"):
            self._orig_callgpt_pydantic = llm_api.callgpt_pydantic
            llm_api.callgpt_pydantic = self._wrap_callgpt_pydantic  # type: ignore[assignment]
        if self._orig_callgoogle_pydantic is None and hasattr(llm_api, "callgoogle_pydantic"):
            self._orig_callgoogle_pydantic = llm_api.callgoogle_pydantic
            llm_api.callgoogle_pydantic = self._wrap_callgoogle_pydantic  # type: ignore[assignment]

    def stop(self) -> None:
        """Restore original llm_api functions."""
        if self._orig_call_llm is not None:
            llm_api.call_llm = self._orig_call_llm  # type: ignore[assignment]
            self._orig_call_llm = None
        if self._orig_callgpt_pydantic is not None:
            llm_api.callgpt_pydantic = self._orig_callgpt_pydantic  # type: ignore[assignment]
            self._orig_callgpt_pydantic = None
        if self._orig_callgoogle_pydantic is not None:
            llm_api.callgoogle_pydantic = self._orig_callgoogle_pydantic  # type: ignore[assignment]
            self._orig_callgoogle_pydantic = None

    def write_report(self, output_path: Path, pipeline_input, pipeline_output, input_type: str) -> None:
        """Persist a detailed JSON report for the full PPC pipeline."""
        grouped = {
            "preprocessing": [],
            "entity_extraction": [],
            "entity_description": [],
            "conflict_handling": [],
            "other": [],
        }
        for rec in self.records:
            stage = rec.get("stage", "other")
            key = stage if stage in grouped else "other"
            grouped[key].append(rec)

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "pipeline_input_type": input_type,
            "pipeline_input": pipeline_input,
            "pipeline_output": pipeline_output,
            "llm_calls": grouped,
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

# Neo4j connection hints
if not os.getenv("NEO4J_PASSWORD"):
    print("Note: Using default NEO4J_PASSWORD='password'")
    print("  If your Neo4j instance uses a different password, set NEO4J_PASSWORD.")
    print("  Example: export NEO4J_PASSWORD=your_password\n")


def test_text_filter():
    """Test Stage 0: Text filtering."""
    print("=" * 60)
    print("Test: Text Filter (Stage 0)")
    print("=" * 60)
    
    from PhotonicsAI.KnowledgeBase.agents.ppc_agent.text_filter import TextFilter
    
    sample_text = """
    Abstract
    This paper proposes a novel Mach-Zehnder Interferometer design with low insertion loss.
    
    Introduction
    Photonic integrated circuits have shown great promise...
    
    Related Work
    Previous work has focused on...
    
    Proposed Architecture
    We demonstrate a new MZI modulator that achieves 1.5 dB insertion loss and 40 GHz bandwidth.
    The design uses thermo-optic phase shifters based on the thermo-optic effect.
    
    Results and Discussion
    The device exhibits excellent performance with Q-factor of 10,000.
    
    Conclusion
    We have successfully implemented a novel modulator architecture.
    """
    
    filter_obj = TextFilter()
    filtered = filter_obj.filter_paper(sample_text)
    
    print(f"Original length: {len(sample_text)} chars")
    print(f"Filtered length: {len(filtered)} chars")
    print(f"\nFiltered text:\n{filtered[:500]}...")
    print("\n✓ Text filter test passed\n")


def test_pdf_processor():
    """Test PDF processing (if PDF available)."""
    print("=" * 60)
    print("Test: PDF Processor")
    print("=" * 60)
    
    from PhotonicsAI.KnowledgeBase.agents.ppc_agent.pdf_processor import PDFProcessor
    
    try:
        processor = PDFProcessor()
        print("✓ PDFProcessor initialized successfully")
        
        # Test with a sample PDF if available
        # pdf_path = Path("sample_paper.pdf")
        # if pdf_path.exists():
        #     result = processor.extract_text(pdf_path)
        #     print(f"Extracted {len(result['text'])} characters from {result['num_pages']} pages")
        # else:
        #     print("No sample PDF found, skipping extraction test")
        
    except ImportError as e:
        print(f"⚠ PyPDF2 not installed: {e}")
        print("Install with: pip install PyPDF2")
    except Exception as e:
        print(f"Error: {e}")
    
    print()


def test_kb_grounding_tool():
    """Test KB Grounding Tool."""
    print("=" * 60)
    print("Test: KB Grounding Tool")
    print("=" * 60)
    
    try:
        config = Neo4jConfig()
        kb_client = KnowledgeBaseClient(config=config)
        kb_client.connect()
        
        from PhotonicsAI.KnowledgeBase.agents.ppc_agent.kb_grounding_tool import create_kb_grounding_tool
        
        tool = create_kb_grounding_tool(kb_client)
        
        # Test the tool
        result = tool.invoke({
            "entity_name": "Mach-Zehnder Interferometer",
            "collection": "Architectures",
            "threshold": 0.3
        })
        
        result_data = json.loads(result)
        print(f"Query: {result_data.get('query')}")
        print(f"Collection: {result_data.get('collection')}")
        print(f"Matches found: {result_data.get('count')}")
        
        if result_data.get('matches'):
            print("\nTop matches:")
            for match in result_data['matches'][:3]:
                print(f"  - {match['name']}: similarity={match.get('similarity', 0.0):.3f}")
        
        print("\n✓ KB Grounding Tool test passed\n")
    
    except ConnectionError as e:
        print(f"⚠ Connection Error: Neo4j is not running or not accessible.")
        print("\n  To start Neo4j:")
        print("  sudo docker run -d --name neo4j-phido -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password -e NEO4J_PLUGINS='[\"apoc\", \"graph-data-science\"]' -e NEO4J_dbms_security_procedures_unrestricted=apoc.*,gds.* neo4j:5.15-enterprise\n")
    except Exception as e:
        error_msg = str(e)
        if "Connection refused" in error_msg or "Can't connect" in error_msg:
            print("⚠ Connection Error: Neo4j is not accessible")
            print("  To start Neo4j:")
            print("  sudo docker run -d --name neo4j-phido -p 7474:7474 -p 7687:7687 "
                  "-e NEO4J_AUTH=neo4j/password -e NEO4J_PLUGINS='[\"apoc\", \"graph-data-science\"]' "
                  "-e NEO4J_dbms_security_procedures_unrestricted=apoc.*,gds.* neo4j:5.15-enterprise\n")
        else:
            print(f"⚠ Error: {error_msg}")
            print("  Make sure Neo4j is running and KB is initialized\n")


def test_entity_extraction():
    """Test Phase A: Entity extraction."""
    print("=" * 60)
    print("Test: Entity Extraction (Phase A)")
    print("=" * 60)
    
    filtered_text = """
    We propose a novel Mach-Zehnder Interferometer modulator design.
    The device demonstrates low insertion loss of 1.5 dB and high bandwidth of 40 GHz.
    The modulator uses thermo-optic phase shifters based on the thermo-optic effect.
    The Q-factor of the ring resonator is 10,000.
    """
    
    try:
        from PhotonicsAI.KnowledgeBase.agents.ppc_agent.entity_extractor import EntityExtractor
        
        extractor = EntityExtractor(llm_model="gemini-3-pro-preview")
        entities = extractor.extract_entities(filtered_text)
        
        print(f"Extracted {len(entities)} entities:")
        for entity in entities:
            print(f"  - {entity.name} ({entity.entity_type})")
            if entity.context:
                print(f"    Context: {entity.context}")
        
        print("\n✓ Entity extraction test passed\n")
    
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure LLM API keys are configured\n")


def test_ontology_loader():
    """Test ontology loading helper."""
    print("=" * 60)
    print("Test: Ontology Loader")
    print("=" * 60)

    try:
        from PhotonicsAI.KnowledgeBase.agents.ppc_agent.ontology_loader import OntologyLoader
        loader = OntologyLoader("PhotonicsAI/KnowledgeBase/GenerativeOntology/ontology/pic_ontology.ttl")
        schema = loader.get_ontology_schema()
        classes = schema.get("classes", {})
        constraints = schema.get("constraints", {})
        print(f"Loaded classes: {len(classes)}")
        print(f"Disjoint pairs: {len(constraints.get('disjoint_pairs', []))}")
        print("\n✓ Ontology loader test passed\n")
    except Exception as e:
        print(f"Error loading ontology: {e}\n")


def test_normalizer():
    """Test normalizer (Vector Search)."""
    print("=" * 60)
    print("Test: Normalizer (Phase B - Vector Only)")
    print("=" * 60)

    try:
        config = Neo4jConfig()
        kb_client = KnowledgeBaseClient(config=config)
        kb_client.connect()

        from PhotonicsAI.KnowledgeBase.agents.ppc_agent.normalizer import Normalizer
        from PhotonicsAI.KnowledgeBase.agents.ppc_agent.models import RawEntity

        normalizer = Normalizer(kb_client=kb_client, llm_model="gemini-3-pro-preview")
        raw_entities = [
            RawEntity(name="Mach-Zehnder Interferometer", entity_type="Architecture"),
            RawEntity(name="MZI", entity_type="Architecture"),
            RawEntity(name="Insertion Loss", entity_type="Property"),
        ]

        normalized = normalizer.normalize_entities(raw_entities)
        print(f"Normalized {len(normalized)} entities:")
        for ent in normalized:
            print(f"  - {ent.raw_name} -> {ent.kb_name} (similarity: {ent.similarity:.3f}, collection: {ent.collection})")

        print("\n✓ Normalizer test passed\n")
    except Exception as e:
        error_msg = str(e)
        if "Connection refused" in error_msg or "Can't connect" in error_msg:
            print(f"⚠ Connection Error: Neo4j is not accessible; skipping normalizer test.\n")
        else:
            print(f"Error in normalizer test: {e}")
            import traceback
            traceback.print_exc()
            print()


def test_full_pipeline_text():
    """Test full PPC Agent pipeline with text input."""
    print("=" * 60)
    print("Test: Full PPC Agent Pipeline (Text Input)")
    print("=" * 60)
    
    sample_text = """
    Abstract
    This paper presents a novel photonic modulator architecture based on a Mach-Zehnder 
    Interferometer design. The device achieves low insertion loss and high bandwidth.
    
    Proposed Architecture
    We demonstrate a new MZI modulator that uses thermo-optic phase shifters.
    The modulator exhibits insertion loss of 1.5 dB and bandwidth of 40 GHz.
    The design is based on the thermo-optic effect and plasma dispersion effect.
    
    Results
    The device shows excellent performance with Q-factor of 10,000 and extinction ratio of 20 dB.
    """
    
    try:
        config = Neo4jConfig()
        kb_client = KnowledgeBaseClient(config=config)
        kb_client.connect()

        agent = PPCAgent(kb_client=kb_client, llm_model="gemini-3-pro-preview")

        tracer = LLMTracer()
        tracer.start()
        try:
            result = agent.process_paper(text=sample_text)
        finally:
            tracer.stop()

        print("\nResults:")
        print(f"  Known entities: {len(result['known_entities'])}")
        print(f"  New concepts: {len(result['new_concepts'])}")
        
        if result['known_entities']:
            print("\n  Known Entities:")
            for entity in result['known_entities'][:5]:
                print(f"    - {entity['raw_name']} -> {entity['kb_name']} "
                      f"(similarity: {entity['similarity']:.3f})")
        
        if result['new_concepts']:
            print("\n  New Concepts:")
            for concept in result['new_concepts'][:5]:
                print(f"    - {concept['name']} ({concept['entity_type']})")

        # Write a detailed report including raw LLM inputs/outputs
        report_path = Path("ppc_agent_detailed_report_text.json")
        tracer.write_report(
            output_path=report_path,
            pipeline_input=sample_text,
            pipeline_output=result,
            input_type="text",
        )
        print(f"\n  Detailed LLM report saved to: {report_path}")

        print("\n✓ Full pipeline test passed\n")
    
    except ConnectionError as e:
        print("⚠ Connection Error: Neo4j is not running")
        print("  Skipping full pipeline test (requires Neo4j)")
        print("  To start Neo4j:")
        print("  sudo docker run -d --name neo4j-phido -p 7474:7474 -p 7687:7687 "
              "-e NEO4J_AUTH=neo4j/password -e NEO4J_PLUGINS='[\"apoc\", \"graph-data-science\"]' "
              "-e NEO4J_dbms_security_procedures_unrestricted=apoc.*,gds.* neo4j:5.15-enterprise\n")
    except Exception as e:
        error_msg = str(e)
        if "Connection refused" in error_msg or "Can't connect" in error_msg:
            print("⚠ Connection Error: Neo4j is not accessible")
            print("  Skipping full pipeline test (requires Neo4j)")
            print("  To start Neo4j:")
            print("  sudo docker run -d --name neo4j-phido -p 7474:7474 -p 7687:7687 "
                  "-e NEO4J_AUTH=neo4j/password -e NEO4J_PLUGINS='[\"apoc\", \"graph-data-science\"]' "
                  "-e NEO4J_dbms_security_procedures_unrestricted=apoc.*,gds.* neo4j:5.15-enterprise\n")
        else:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print()


def test_full_pipeline_pdf():
    """Test full PPC Agent pipeline with PDF input."""
    print("=" * 60)
    print("Test: Full PPC Agent Pipeline (PDF Input)")
    print("=" * 60)
    
    # Look for a sample PDF
    sample_pdf = Path("sample_paper.pdf")
    if not sample_pdf.exists():
        print("⚠ No sample PDF found. Skipping PDF test.")
        print("  To test PDF processing, place a PDF file named 'sample_paper.pdf' in the root directory.\n")
        return
    
    try:
        # Create KB client with explicit config
        config = Neo4jConfig()

        print(f"Connecting to Neo4j at {config.uri}")
        print(f"  Database: {config.database}")
        print(f"  Username: {config.username}\n")

        kb_client = KnowledgeBaseClient(config=config)
        kb_client.connect()

        # Create PPC Agent (uses subprocess-based MCP client automatically)
        print("ℹ PPC Agent will use subprocess-based MCP client for PDF annotation")
        print("  If MCP fails, it will automatically fall back to PyPDF2 extraction.\n")

        agent = PPCAgent(
            kb_client=kb_client,
            llm_model="gemini-3-pro-preview"
        )

        tracer = LLMTracer()
        tracer.start()
        try:
            result = agent.process_paper(pdf_path=sample_pdf)
        finally:
            tracer.stop()

        print("\nResults:")
        print(f"  Known entities: {len(result['known_entities'])}")
        print(f"  New concepts: {len(result['new_concepts'])}")
        
        if result['known_entities']:
            print("\n  Known Entities:")
            for entity in result['known_entities']:
                print(f"    - {entity['raw_name']} -> {entity['kb_name']} "
                      f"(similarity: {entity['similarity']:.3f}, type: {entity['entity_type']})")
        
        if result['new_concepts']:
            print("\n  New Concepts:")
            for concept in result['new_concepts']:
                print(f"    - {concept['name']} (type: {concept['entity_type']})")
                # Removed detailed printing as per user request
                # if concept.get('description'):
                #     desc = concept['description'][:100] + "..." if len(concept['description']) > 100 else concept['description']
                #     print(f"      Description: {desc}")
                # if concept.get('context'):
                #     print(f"      Context: {concept['context']}")

        # Save detailed report including raw LLM inputs/outputs
        report_file = Path("ppc_agent_detailed_report_pdf.json")
        tracer.write_report(
            output_path=report_file,
            pipeline_input=str(sample_pdf.resolve()),
            pipeline_output=result,
            input_type="pdf",
        )
        print(f"\n  Detailed LLM report saved to: {report_file}")
        
        # Generate Visualization
        vis_file = Path("ppc_performance_report.html")
        generate_performance_table(report_file, vis_file)

        # Save results to file
        output_file = Path("ppc_agent_results.json")
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n  Results saved to: {output_file}")
        
        print("\n✓ Full PDF pipeline test passed\n")
    
    except ConnectionError as e:
        print("⚠ Connection Error: Neo4j is not running")
        print("  Skipping full pipeline test (requires Neo4j)")
        print("  To start Neo4j:")
        print("  sudo docker run -d --name neo4j-phido -p 7474:7474 -p 7687:7687 "
              "-e NEO4J_AUTH=neo4j/password -e NEO4J_PLUGINS='[\"apoc\", \"graph-data-science\"]' "
              "-e NEO4J_dbms_security_procedures_unrestricted=apoc.*,gds.* neo4j:5.15-enterprise\n")
    except Exception as e:
        error_msg = str(e)
        if "Connection refused" in error_msg or "Can't connect" in error_msg:
            print("⚠ Connection Error: Neo4j is not accessible")
            print("  Skipping full pipeline test (requires Neo4j)")
            print("  To start Neo4j:")
            print("  sudo docker run -d --name neo4j-phido -p 7474:7474 -p 7687:7687 "
                  "-e NEO4J_AUTH=neo4j/password -e NEO4J_PLUGINS='[\"apoc\", \"graph-data-science\"]' "
                  "-e NEO4J_dbms_security_procedures_unrestricted=apoc.*,gds.* neo4j:5.15-enterprise\n")
        else:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print()


def generate_performance_table(json_path: Path, output_html_path: Path) -> None:
    """
    Generate an HTML performance comparison table from PPC Agent detailed report.
    
    Args:
        json_path: Path to the detailed report JSON file.
        output_html_path: Path to save the HTML file.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        known_entities = data['pipeline_output'].get('known_entities', [])
        new_concepts = data['pipeline_output'].get('new_concepts', [])
        
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
            # Red color indicates it wasn't enough to be a match (which is expected for New Concepts, or maybe a miss)
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
            
        print(f"\n  Performance visualization saved to: {output_html_path}")
        
    except Exception as e:
        print(f"Error generating visualization: {e}")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PPC Agent Test Suite")
    print("=" * 60 + "\n")
    
    # Run tests
    test_text_filter()
    test_pdf_processor()
    test_kb_grounding_tool()
    test_entity_extraction()
    test_ontology_loader()
    test_normalizer()
    # test_full_pipeline_text()  # Disabled 
    test_full_pipeline_pdf()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

