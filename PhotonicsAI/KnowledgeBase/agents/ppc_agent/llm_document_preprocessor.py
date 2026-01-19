"""LLM-based document preprocessor for identifying relevant sections.

This module uses an LLM to analyze the full document and identify sections
that discuss photonic components, architectures, and related technical content.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from PhotonicsAI.Photon import llm_api


class LLMDocumentPreprocessor:
    """Preprocess documents using LLM to identify relevant sections."""
    
    def __init__(self, llm_model: str = "gemini-2.5-pro"):
        """
        Initialize LLM document preprocessor.
        
        Args:
            llm_model: LLM model to use for section identification
        """
        self.llm_model = llm_model
    
    def preprocess_pdf(self, pdf_path: Path) -> str:
        """
        Preprocess a PDF by extracting text and identifying relevant sections.
        
        The preferred path is:
          1. Use AxDocumentParser MCP tool to convert the PDF into structured markdown.
          2. Feed the markdown content into the LLM to isolate relevant sections.
        
        If the MCP tool is unavailable or fails, falls back to local PDF text extraction.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted relevant text sections as a single string
        """
        # First, try AxDocumentParser MCP tool to get structured markdown
        full_text: Optional[str] = None
        try:
            from .mcp_client_subprocess import SubprocessMCPClient
            client = SubprocessMCPClient()
            try:
                print("Preprocess PDF: Using AxDocumentParser MCP tool for markdown conversion...")
                full_text = client.parse_pdf_to_markdown(str(pdf_path))
                print(f"  Debug: AxDocumentParser output length: {len(full_text)} characters")
            finally:
                client.close()
        except Exception as mcp_error:
            print(f"ℹ AxDocumentParser MCP preprocessing failed ({mcp_error}). Falling back to local PDF text extraction.")
        
        # Fallback: local PDF processing if MCP path failed or returned empty
        if not full_text or not full_text.strip():
            from .pdf_processor import PDFProcessor
            pdf_processor = PDFProcessor()
            pdf_data = pdf_processor.extract_text(pdf_path)
            full_text = pdf_data["text"]
        
        if not full_text.strip():
            raise ValueError(f"Could not extract text from PDF (MCP + fallback failed): {pdf_path}")
        
        # Use LLM to identify relevant sections
        relevant_sections = self._identify_relevant_sections(full_text)
        
        # Combine relevant sections into a single text
        combined_text = self._combine_sections(relevant_sections)
        
        # Post-process: Rename extracted figures to match captions using LLM
        if full_text:
            try:
                # We need to process the FULL text (which contains the markdown images) to do the renaming
                # self._rename_figures_by_caption_llm(full_text, pdf_path)
                pass # Feature disabled per user request (unreliable)
            except Exception as e:
                print(f"Warning: Figure renaming failed: {e}")
                
        return combined_text
    
    def extract_figure_captions(self, markdown_text: str) -> Dict[str, str]:
        """
        Extract mapping between figure filenames and their captions from markdown.
        Relies on the sequential output of AxDocumentParser (file order usually matches caption order).
        
        Args:
            markdown_text: The full markdown text containing image links and captions.
            
        Returns:
            Dict mapping 'filename.png' -> 'Full Caption Text'
        """
        import re
        
        # Regex to capture image links and their subsequent text (likely caption)
        # Matches: ![alt](path) ... text up to next double newline
        # Lazy match for text to capture the immediate caption paragraph
        pattern = re.compile(r'!\[.*?\]\((.*?)\)\s*(.*?)(?=\n\n|\Z)', re.DOTALL)
        
        captions = {}
        for match in pattern.finditer(markdown_text):
            filename = match.group(1).strip()
            caption_text = match.group(2).strip()
            
            # Clean up caption: remove bolding, extra whitespace
            caption_text = caption_text.replace('**', '').replace('\n', ' ')
            
            # Only keep if it looks like a caption (starts with Fig/Figure)
            if re.match(r'^(?:Fig\.?|Figure)\s*\d+', caption_text, re.IGNORECASE):
                captions[filename] = caption_text
                
        return captions
    
    def _rename_figures_by_caption_llm(self, markdown_text: str, pdf_path: Path) -> None:
        """
        Use LLM to identify the mapping between image files and their semantic figure labels (e.g., Fig. 1, Fig. 2a).
        Then rename the files on disk.
        """
        import shutil
        import json
        
        if not markdown_text:
            return

        # 1. Extract candidate image links to keep the prompt small
        # We look for lines containing "![...](...)"
        image_lines = []
        lines = markdown_text.split('\n')
        for i, line in enumerate(lines):
            if "![" in line and "](" in line:
                # Get a small window of context around the image (captions are usually nearby)
                start = max(0, i - 2)
                end = min(len(lines), i + 5)
                context_window = "\n".join(lines[start:end])
                image_lines.append(context_window)
        
        if not image_lines:
            return

        context_str = "\n---\n".join(image_lines)
        
        # 2. Ask LLM to map filenames to Figure labels
        sys_prompt = """You are an assistant helping to organize figures from a research paper.
Your task is to identify the correct Figure Label (e.g., "Fig_1", "Fig_2a") for each image file based on its surrounding text context (captions).

Rules:
1. Look for the Caption immediately following or preceding the image link.
2. Extract the Figure Number (e.g., "Fig. 1", "Figure 2", "Fig. 3(a)").
3. Format the target label as: "Fig_{number}" or "Fig_{number}{letter}" (e.g., Fig_1, Fig_2a).
4. Ignore generic images without clear figure captions.
5. Return a JSON mapping: {"original_filename.png": "Fig_X"}
"""
        
        prompt = f"""Identify the Figure Labels for the images in the following text snippets.
        
Snippets:
{context_str}

Return JSON format:
{{
  "mappings": {{
    "sample_paper_fig_1.png": "Fig_1",
    "sample_paper_fig_5.png": "Fig_2a"
  }}
}}
"""
        try:
            response = llm_api.call_llm(prompt, sys_prompt, self.llm_model)
            
            # Parse JSON
            json_str = self._extract_json_from_text(response) # Reuse existing helper or simplistic parse
            if not json_str:
                return
                
            mapping_data = json.loads(json_str)
            mappings = mapping_data.get("mappings", {})
            
            # 3. Rename files
            for old_name, new_label in mappings.items():
                if not old_name or not new_label:
                    continue
                    
                # Clean up filenames
                old_name = old_name.strip()
                new_label = new_label.strip()
                
                # Check CWD first, then PDF dir
                old_file = Path(old_name)
                if not old_file.exists():
                    old_file = pdf_path.parent / old_file.name
                
                if not old_file.exists():
                    print(f"    Debug: Could not find original file {old_name} to rename.")
                    continue
                
                # Construct new filename: {pdf_stem}_{new_label}.png
                # e.g. sample_paper_Fig_1.png
                new_filename = f"{pdf_path.stem}_{new_label}{old_file.suffix}"
                new_file = old_file.parent / new_filename
                
                if old_file != new_file:
                    if new_file.exists():
                        print(f"    Debug: Target file {new_filename} already exists. Skipping.")
                    else:
                        print(f"    Debug: Renaming {old_file.name} -> {new_filename}")
                        shutil.move(str(old_file), str(new_file))
                        
        except Exception as e:
            print(f"    Debug: LLM figure renaming error: {e}")

    def _extract_json_from_text(self, text: str) -> str:
        """Extract JSON substring."""
        import json
        text = text.strip()
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return text[start:end+1]
        return ""

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess raw text by identifying relevant sections.
        
        Args:
            text: Raw text content
            
        Returns:
            Extracted relevant text sections as a single string
        """
        relevant_sections = self._identify_relevant_sections(text)
        return self._combine_sections(relevant_sections)
    
    def _identify_relevant_sections(self, full_text: str) -> List[Dict[str, Any]]:
        """
        Use LLM to identify relevant sections from the full document.
        
        Args:
            full_text: Full document text
            
        Returns:
            List of section dictionaries with 'text', 'section_type', 'reason'
        """
        sys_prompt = """You are an assistant helping to extract relevant technical content from photonics research papers.

Your task is to identify specific sections and paragraphs from the document that discuss:
1. **Proposed architectures and components**: Novel photonic components, integrated circuits, or system architectures
2. **Component specifications**: Technical details about components (modulators, detectors, waveguides, filters, etc.)
3. **Design functions**: What the components/architectures do (modulation, filtering, switching, etc.)
4. **Physical principles**: Underlying mechanisms (electro-optic effect, thermo-optic effect, resonance, etc.)
5. **Properties and performance metrics**: Insertion loss, bandwidth, Q factor, efficiency, extinction ratio, etc.

**Focus on:**
- Abstract and Introduction (thesis/claim)
- Proposed Architecture/Design sections
- Results and Discussion (component descriptions)
- Conclusion (novel contributions summary)

**Exclude:**
- Related Work and Background (unless describing novel components)
- Experimental Setup (unless defining new components)
- References and Acknowledgments

For each relevant section you identify, provide:
- The exact text excerpt
- The section type (e.g., "Abstract", "Proposed Architecture", "Results", "Conclusion")
- A brief reason why it's relevant

Return a JSON list of sections, each with: 'text', 'section_type', 'reason'"""
        
        # Limit to first 50k chars to avoid token limits
        document_text = full_text[:50000] if len(full_text) > 50000 else full_text
        
        prompt = f"""Analyze the following document and identify all sections and paragraphs that discuss photonic components, architectures, design functions, physical principles, or component properties.

Document:
{document_text}

Identify and extract the most relevant sections. For each section, provide:
1. The exact text excerpt (preserve the original wording)
2. The section type (Abstract, Introduction, Proposed Architecture, Results, Conclusion, etc.)
3. A brief reason why it's relevant

Focus on technical content that would be useful for entity extraction."""
        
        # Use LLM to identify sections (using call_llm to respect model parameter)
        print(f"  Debug: Calling LLM with model: {self.llm_model}")
        print(f"  Debug: Prompt length: {len(prompt)} characters")
        response = llm_api.call_llm(prompt, sys_prompt, self.llm_model)
        
        # Check if response is valid
        if response is None:
            raise ValueError(
                f"LLM call returned None. Model '{self.llm_model}' may not be supported by call_llm. "
                f"Supported models: gpt-*, nvidia/*, o1*, o3*, deepseek*, gemini-*, claude-*"
            )
        
        if not isinstance(response, str):
            raise ValueError(f"LLM call returned unexpected type: {type(response)}. Expected string, got {type(response)}.")
        
        if not response.strip():
            raise ValueError("LLM call returned empty response.")
        
        print(f"  Debug: LLM response length: {len(response)} characters")
        
        # Try to parse JSON from response (robust to nested braces / long arrays)
        import json
        import re

        def _strip_code_fences(txt: str) -> str:
            txt = txt.strip()
            # Remove surrounding markdown fences if present
            fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", txt, re.IGNORECASE)
            if fence:
                return fence.group(1).strip()
            return txt

        def _extract_balanced_json(txt: str) -> str:
            """
            Extract the first balanced JSON array/object substring from txt.
            This avoids regex truncation when nested braces/brackets exist.
            """
            s = _strip_code_fences(txt)
            # Prefer array output
            for opener, closer in [("[", "]"), ("{", "}")]:
                start = s.find(opener)
                if start == -1:
                    continue
                depth = 0
                in_str = False
                esc = False
                for i in range(start, len(s)):
                    ch = s[i]
                    if in_str:
                        if esc:
                            esc = False
                        elif ch == "\\":
                            esc = True
                        elif ch == '"':
                            in_str = False
                        continue
                    else:
                        if ch == '"':
                            in_str = True
                            continue
                        if ch == opener:
                            depth += 1
                        elif ch == closer:
                            depth -= 1
                            if depth == 0:
                                return s[start : i + 1]
                # If unbalanced, fall through
            return s

        json_str = _extract_balanced_json(response)
        try:
            sections_data = json.loads(json_str)
        except Exception:
            # Try a second-chance strategy: often the model returns leading text then JSON.
            # Attempt using last '[' ... last ']' to maximize chance of a valid list.
            s = _strip_code_fences(response)
            a, b = s.find("["), s.rfind("]")
            if a != -1 and b != -1 and b > a:
                try:
                    sections_data = json.loads(s[a : b + 1])
                except Exception:
                    sections_data = None
            else:
                sections_data = None

        if isinstance(sections_data, list):
            return sections_data
        if isinstance(sections_data, dict) and "sections" in sections_data and isinstance(sections_data["sections"], list):
            return sections_data["sections"]

        # Last resort: create a single section from the response
        return [{
            "text": response,
            "section_type": "LLM Analysis",
            "reason": "LLM response (JSON parsing failed)"
        }]
    
    def _combine_sections(self, sections: List[Dict[str, Any]]) -> str:
        """
        Combine identified sections into a single text string.
        
        Args:
            sections: List of section dictionaries
            
        Returns:
            Combined text with section markers
        """
        if not sections:
            return ""
        
        combined_parts = []
        for i, section in enumerate(sections, 1):
            section_type = section.get("section_type", "Unknown")
            text = section.get("text", "")
            reason = section.get("reason", "")
            
            # Add section marker
            combined_parts.append(f"[Section {i}: {section_type}]")
            if reason:
                combined_parts.append(f"Reason: {reason}")
            combined_parts.append("")
            combined_parts.append(text)
            combined_parts.append("")
        
        return "\n".join(combined_parts)

