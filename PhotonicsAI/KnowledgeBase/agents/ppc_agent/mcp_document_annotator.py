"""MCP-based document annotation using AxDocumentAnnotator for PDF processing."""

import os
from pathlib import Path
from typing import Dict, Optional, Any

# Import subprocess-based MCP client (reliable, bypasses SDK issues)
try:
    from .mcp_client_subprocess import SubprocessMCPClient
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    SubprocessMCPClient = None


class MCPDocumentAnnotator:
    """Extract relevant text from PDFs using AxDocumentAnnotator MCP tool.
    
    This class uses the subprocess-based MCP client to extract relevant
    sections and context from PDF papers about photonic components and architectures.
    """
    
    # Default query for extracting photonic components and architectures
    DEFAULT_QUERY = """Extract all sections and context from this paper that describe:
1. Photonic components (e.g., modulators, detectors, waveguides, couplers, resonators, filters)
2. Photonic architectures and systems (e.g., interferometers, ring resonators, photonic circuits)
3. Component properties and specifications (e.g., insertion loss, bandwidth, Q factor, efficiency, extinction ratio)
4. Design functions and capabilities (e.g., modulation, filtering, switching, routing)
5. Physical principles and mechanisms (e.g., electro-optic effect, thermo-optic effect, resonance, interference)

Focus on:
- Abstract and Introduction sections
- Proposed Architecture/Design sections
- Results and Discussion sections that describe components
- Conclusion sections that summarize novel contributions

Exclude:
- Related Work and Background sections (unless they describe novel components)
- Experimental Setup sections (unless they define new components)
- References and Acknowledgments

Provide the extracted text with clear section markers and context about what each component/architecture does."""
    
    def __init__(self, query: Optional[str] = None):
        """
        Initialize MCP Document Annotator.
        
        Args:
            query: Custom query for annotation. If None, uses DEFAULT_QUERY.
        """
        self.query = query or self.DEFAULT_QUERY
        self._client: Optional[SubprocessMCPClient] = None
    
    def _get_client(self) -> SubprocessMCPClient:
        """Get or create MCP client instance."""
        if not MCP_AVAILABLE:
            raise RuntimeError(
                "MCP client not available. The subprocess MCP client module could not be imported."
            )
        
        if self._client is None:
            self._client = SubprocessMCPClient()
        
        return self._client
    
    def annotate_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Annotate a PDF file using AxDocumentAnnotator MCP tool.
        
        Args:
            pdf_path: Path to PDF file (must be absolute path)
            
        Returns:
            Dictionary with:
            - 'annotated_text': Extracted relevant text (concatenated from all annotations)
            - 'full_annotation': Complete annotation response (may be structured)
            - 'source': Source identifier ('mcp_annotator')
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Ensure absolute path
        pdf_path = pdf_path.resolve()
        
        # Get MCP client
        client = self._get_client()
        
        # Call MCP tool
        try:
            annotation_result = client.annotate_file(
                file_path=str(pdf_path),
                query=self.query
            )
            
            # Process the annotation result
            # The result may be a string or structured object
            # Debug: show what we received
            if isinstance(annotation_result, str):
                print(f"  Debug: Annotation result type: string, length: {len(annotation_result)}")
                print(f"  Debug: First 200 chars: {annotation_result[:200]}")
                print(f"  Debug: Last 200 chars: {annotation_result[-200:]}")
                # Check if it contains annotation markers
                if "**Annotation" in annotation_result or "Description:" in annotation_result:
                    print(f"  Debug: Found annotation markers in result")
                if "No annotations found" in annotation_result:
                    print(f"  Debug: Warning: 'No annotations found' message detected")
            else:
                print(f"  Debug: Annotation result type: {type(annotation_result)}")
            
            # Check if the result mentions a saved markdown file
            # The MCP tool may return a status message with the file path
            markdown_file_path = None
            if isinstance(annotation_result, str):
                import re
                # Look for "saved markdown file: /path/to/file.md" pattern
                match = re.search(r'saved markdown file:\s*(.+?\.md)', annotation_result, re.IGNORECASE)
                if match:
                    markdown_file_path = Path(match.group(1).strip())
                    print(f"  Debug: Found markdown file path in result: {markdown_file_path}")
            
            # If we found a markdown file path and it exists, read it
            if markdown_file_path and markdown_file_path.exists():
                print(f"  Debug: Reading annotations from markdown file: {markdown_file_path}")
                with open(markdown_file_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
                annotated_text = self._extract_text_from_annotation(markdown_content)
            else:
                # Extract from the annotation result directly
                annotated_text = self._extract_text_from_annotation(annotation_result)
            
            print(f"  Debug: Extracted text length: {len(annotated_text)}")
            if len(annotated_text) < 500:
                print(f"  Debug: Extracted text preview: {annotated_text[:300]}")
            
            return {
                "annotated_text": annotated_text,
                "full_annotation": annotation_result,
                "source": "mcp_annotator"
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to annotate PDF with MCP tool: {e}")
        finally:
            # Clean up client
            if self._client:
                self._client.close()
                self._client = None
    
    def _extract_text_from_annotation(self, annotation_result: Any) -> str:
        """
        Extract text content from annotation result.
        
        The annotation result from AxDocumentAnnotator is a formatted string containing:
        - Query information
        - Multiple annotations with descriptions, types, tags, and references
        - Page numbers for PDF files
        
        Format example:
        **Annotation 1** (Page 2):
        Type: AnnotationType.TEXT
        Description: ...
        Tags: ...
        Reference: ...
        
        We extract the description and reference text from each annotation.
        
        Args:
            annotation_result: Result from MCP annotator tool (string or structured object)
            
        Returns:
            Extracted text as string, formatted for entity extraction
        """
        if isinstance(annotation_result, str):
            # The MCP tool returns a formatted string with annotations
            # Extract the useful content from the formatted string
            import re
            
            # First, check if the result contains "No annotations found"
            if "No annotations found" in annotation_result:
                # The annotations might be in the markdown file, or the query didn't match
                # Try to extract any annotation content that might be in the result anyway
                pass  # Continue with extraction below
            
            extracted_parts = []
            
            # Method 1: Extract Description and Reference fields using regex
            # Pattern matches: Description: ... (until Tags: or Reference: or Type: or end of annotation)
            descriptions = re.findall(
                r'Description:\s*(.+?)(?=\n\s*(?:Tags:|Reference:|Type:|$|\*\*Annotation))',
                annotation_result,
                re.DOTALL | re.MULTILINE
            )
            
            # Pattern matches: Reference: ... (until Description: or Type: or Tags: or end of annotation)
            references = re.findall(
                r'Reference:\s*(.+?)(?=\n\s*(?:Description:|Type:|Tags:|$|\*\*Annotation))',
                annotation_result,
                re.DOTALL | re.MULTILINE
            )
            
            # Combine descriptions and references
            for desc in descriptions:
                desc_clean = desc.strip()
                if desc_clean:
                    extracted_parts.append(desc_clean)
            
            for ref in references:
                ref_clean = ref.strip()
                if ref_clean:
                    extracted_parts.append(ref_clean)
            
            # Method 2: If regex didn't work, try line-by-line parsing
            if not extracted_parts:
                lines = annotation_result.split('\n')
                current_description = None
                current_reference = None
                in_annotation = False
                
                for line in lines:
                    line = line.strip()
                    
                    # Skip empty lines and headers
                    if not line:
                        continue
                    
                    # Detect annotation start
                    if line.startswith('**Annotation') or (line.startswith('Annotation') and 'Page' in line):
                        # Save previous annotation
                        if current_description:
                            extracted_parts.append(current_description)
                        if current_reference:
                            extracted_parts.append(current_reference)
                        in_annotation = True
                        current_description = None
                        current_reference = None
                        continue
                    
                    # Skip Type and Tags lines
                    if line.startswith('Type:') or line.startswith('Tags:'):
                        continue
                    
                    # Extract description
                    if line.startswith('Description:'):
                        current_description = line.replace('Description:', '').strip()
                    elif current_description is None and 'Description:' in line:
                        current_description = line.split('Description:')[-1].strip()
                    elif in_annotation and current_description is None and len(line) > 10:
                        # Might be description on next line
                        current_description = line
                    
                    # Extract reference
                    elif line.startswith('Reference:'):
                        current_reference = line.replace('Reference:', '').strip()
                    elif 'Reference:' in line:
                        current_reference = line.split('Reference:')[-1].strip()
                    elif in_annotation and current_reference is None and current_description and len(line) > 20:
                        # Might be reference on next line
                        current_reference = line
                
                # Add the last annotation
                if current_description:
                    extracted_parts.append(current_description)
                if current_reference:
                    extracted_parts.append(current_reference)
            
            # If we extracted content, return it
            if extracted_parts:
                print(f"  Debug: Extracted {len(extracted_parts)} parts from regex")
                return "\n\n".join(extracted_parts)
            else:
                print(f"  Debug: Regex extraction found 0 descriptions and 0 references")
            
            # Method 3: If still nothing, try to extract all non-header text
            # Remove headers and metadata, keep content
            lines = annotation_result.split('\n')
            content_lines = []
            skip_next = False
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Skip headers
                if line.startswith('**') and ('Query' in line or 'Annotations' in line or 'Annotation' in line):
                    skip_next = True
                    continue
                
                # Skip Type and Tags
                if line.startswith('Type:') or line.startswith('Tags:'):
                    continue
                
                # Skip "Successfully annotated" messages
                if 'Successfully' in line or 'saved markdown' in line.lower():
                    continue
                
                # Skip "No annotations found" if it's the only meaningful content
                if line == "No annotations found for the given query.":
                    # Only skip if this is the only content (very short result)
                    if len(annotation_result) < 100:
                        continue
                    # Otherwise, it might be part of a larger result, so keep it
                
                # Keep Description and Reference lines (with their content)
                if line.startswith('Description:') or line.startswith('Reference:'):
                    content_lines.append(line)
                # Keep annotation headers (they contain page info)
                elif line.startswith('**Annotation') or (line.startswith('Annotation') and 'Page' in line):
                    # Extract page number if available
                    page_match = re.search(r'Page\s+(\d+)', line)
                    if page_match:
                        content_lines.append(f"[Page {page_match.group(1)}]")
                elif len(line) > 20 and not line.startswith('**'):
                    # Likely content (but not a header)
                    content_lines.append(line)
            
            if content_lines:
                print(f"  Debug: Method 3 extracted {len(content_lines)} content lines")
                return "\n\n".join(content_lines)
            
            # Last resort: return the original string (minus status messages)
            # Remove status messages but keep everything else
            cleaned = re.sub(r'Successfully.*?\.md\s*', '', annotation_result, flags=re.DOTALL | re.IGNORECASE)
            cleaned = re.sub(r'Successfully annotated.*?\.pdf\s*', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
            if cleaned.strip() and cleaned.strip() != "No annotations found for the given query.":
                print(f"  Debug: Using cleaned result (removed status messages)")
                return cleaned.strip()
            
            print(f"  Debug: All extraction methods failed, returning original")
            return annotation_result
        
        # If it's a dict or object, try to extract text
        if isinstance(annotation_result, dict):
            # Check for common keys
            if "annotations" in annotation_result:
                # Extract text from each annotation
                texts = []
                for ann in annotation_result.get("annotations", []):
                    if isinstance(ann, dict):
                        # Extract description, text, or content
                        text = (
                            ann.get("description") or
                            ann.get("text") or
                            ann.get("content") or
                            str(ann)
                        )
                        texts.append(text)
                    else:
                        texts.append(str(ann))
                return "\n\n".join(texts)
            elif "text" in annotation_result:
                return annotation_result["text"]
            elif "content" in annotation_result:
                return annotation_result["content"]
            else:
                # Try to stringify the dict
                return str(annotation_result)
        
        # For other types, convert to string
        return str(annotation_result)
    
    def extract_relevant_text(self, pdf_path: Path) -> str:
        """
        Extract relevant text from PDF using MCP annotation.
        
        Convenience method that returns just the annotated text.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted relevant text as string
        """
        result = self.annotate_pdf(pdf_path)
        return result["annotated_text"]

