"""Stage 0: Non-LLM text pre-filter for paper sections and keywords."""

import re
from typing import List, Dict


class TextFilter:
    """Filter paper text to extract relevant sections and paragraphs."""
    
    # Target sections (case-insensitive patterns)
    TARGET_SECTIONS = [
        r'^\s*(abstract|summary)',
        r'^\s*(introduction|intro)',
        r'^\s*(proposed\s+(architecture|design|method|approach|system))',
        r'^\s*(design\s+(and\s+)?implementation)',
        r'^\s*(results?\s+(and\s+)?(discussion|analysis))',
        r'^\s*(conclusion|conclusions|summary)',
        r'^\s*(novel|new|innovative)',
    ]
    
    # Exclude sections
    EXCLUDE_SECTIONS = [
        r'^\s*(related\s+work|prior\s+work|background|literature\s+review)',
        r'^\s*(experimental\s+setup|experimental\s+method|methodology)',
        r'^\s*(acknowledgment|acknowledgement|references|bibliography)',
        r'^\s*(appendix|appendices)',
    ]
    
    # High-value keywords/phrases
    KEY_VERBS = [
        'demonstrate', 'propose', 'achieve', 'implement', 'novel', 'innovative',
        'design', 'develop', 'present', 'show', 'exhibit', 'enable'
    ]
    
    PROPERTY_KEYWORDS = [
        'insertion loss', 'bandwidth', 'q factor', 'q-factor', 'efficiency',
        'extinction ratio', 'phase shift', 'coupling', 'resonance',
        'wavelength', 'frequency', 'power', 'gain', 'noise'
    ]
    
    def __init__(self):
        """Initialize text filter."""
        pass
    
    def filter_paper(self, text: str) -> str:
        """
        Filter paper text to extract relevant sections and paragraphs.
        
        Args:
            text: Full paper text
            
        Returns:
            Filtered text chunk containing only relevant paragraphs
        """
        # Split into sections
        sections = self._identify_sections(text)
        
        # Extract target sections
        target_sections = []
        for section_name, section_text in sections.items():
            if self._is_target_section(section_name):
                target_sections.append(section_text)
        
        # If no sections found, use full text but filter by keywords
        if not target_sections:
            filtered_paragraphs = self._filter_by_keywords(text)
            return "\n\n".join(filtered_paragraphs)
        
        # Combine target sections and filter by keywords
        combined_text = "\n\n".join(target_sections)
        filtered_paragraphs = self._filter_by_keywords(combined_text)
        
        return "\n\n".join(filtered_paragraphs)
    
    def _identify_sections(self, text: str) -> Dict[str, str]:
        """
        Identify sections in the paper based on headings.
        
        Returns:
            Dictionary mapping section names to section text
        """
        sections = {}
        lines = text.split('\n')
        
        current_section = "Introduction"  # Default section
        current_text = []
        
        for line in lines:
            # Check if line looks like a heading (short, capitalized, possibly numbered)
            stripped = line.strip()
            if self._is_heading(stripped):
                # Save previous section
                if current_text:
                    sections[current_section] = "\n".join(current_text)
                
                # Start new section
                current_section = stripped
                current_text = []
            else:
                current_text.append(line)
        
        # Save last section
        if current_text:
            sections[current_section] = "\n".join(current_text)
        
        return sections
    
    def _is_heading(self, line: str) -> bool:
        """Check if a line looks like a section heading."""
        if not line or len(line) > 100:
            return False
        
        # Check for common heading patterns
        # All caps short line
        if line.isupper() and len(line.split()) <= 10:
            return True
        
        # Numbered heading (e.g., "1. Introduction", "2.1 Background")
        if re.match(r'^\d+\.?\s+[A-Z]', line):
            return True
        
        # Roman numerals (e.g., "I. Introduction")
        if re.match(r'^[IVX]+\.\s+[A-Z]', line):
            return True
        
        # Check against known section patterns
        line_lower = line.lower()
        for pattern in self.TARGET_SECTIONS + self.EXCLUDE_SECTIONS:
            if re.match(pattern, line_lower, re.IGNORECASE):
                return True
        
        return False
    
    def _is_target_section(self, section_name: str) -> bool:
        """Check if section should be included."""
        section_lower = section_name.lower()
        
        # Check exclude patterns first
        for pattern in self.EXCLUDE_SECTIONS:
            if re.search(pattern, section_lower, re.IGNORECASE):
                return False
        
        # Check target patterns
        for pattern in self.TARGET_SECTIONS:
            if re.search(pattern, section_lower, re.IGNORECASE):
                return True
        
        return False
    
    def _filter_by_keywords(self, text: str) -> List[str]:
        """
        Filter paragraphs by keywords and phrases.
        
        Returns:
            List of relevant paragraphs
        """
        paragraphs = text.split('\n\n')
        relevant_paragraphs = []
        
        for para in paragraphs:
            if not para.strip():
                continue
            
            para_lower = para.lower()
            
            # Check for architecture/component names (capitalized, hyphenated)
            has_component_name = bool(re.search(r'\b[A-Z][a-z]+(-[A-Z][a-z]+)+\b', para))
            
            # Check for key verbs
            has_key_verb = any(verb in para_lower for verb in self.KEY_VERBS)
            
            # Check for property keywords
            has_property = any(prop in para_lower for prop in self.PROPERTY_KEYWORDS)
            
            # Include if has any of these indicators
            if has_component_name or has_key_verb or has_property:
                relevant_paragraphs.append(para)
        
        return relevant_paragraphs

