"""PDF text and image extraction using PyPDF2."""

import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None


class PDFProcessor:
    """Extract text and images from PDF files."""
    
    def __init__(self):
        """Initialize PDF processor."""
        if PyPDF2 is None:
            raise ImportError("PyPDF2 is required. Install with: pip install PyPDF2")
    
    def extract_text(self, pdf_path: Path) -> Dict[str, any]:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with 'text' (full text), 'pages' (list of page texts),
            and 'images' (list of image streams)
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        text_pages = []
        images = []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                # Extract text
                page_text = page.extract_text()
                text_pages.append({
                    "page": page_num + 1,
                    "text": page_text
                })
                
                # Extract images (if any)
                if '/XObject' in page['/Resources']:
                    xobjects = page['/Resources']['/XObject'].get_object()
                    for obj_name in xobjects:
                        obj = xobjects[obj_name]
                        if obj['/Subtype'] == '/Image':
                            try:
                                # Get image data
                                image_data = obj.get_data()
                                images.append({
                                    "page": page_num + 1,
                                    "name": obj_name,
                                    "data": image_data,
                                    "width": obj.get('/Width'),
                                    "height": obj.get('/Height')
                                })
                            except Exception as e:
                                # Skip images that can't be extracted
                                continue
        
        full_text = "\n\n".join([p["text"] for p in text_pages])
        
        return {
            "text": full_text,
            "pages": text_pages,
            "images": images,
            "num_pages": len(text_pages)
        }
    
    def extract_text_simple(self, pdf_path: Path) -> str:
        """
        Extract text from PDF file (simple version, no images).
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Full text as string
        """
        result = self.extract_text(pdf_path)
        return result["text"]

