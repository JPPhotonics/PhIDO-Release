"""Acronym Agent for resolving and managing acronym mappings."""

import json
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

from PhotonicsAI.Photon import llm_api


class AcronymAgent:
    """
    Agent responsible for maintaining a persistent acronym mapping and resolving
    unknown acronyms using an LLM.
    """

    def __init__(self, mapping_file: str = "acronym_mappings.json", llm_model: str = "gemini-2.5-pro"):
        """
        Initialize AcronymAgent.

        Args:
            mapping_file: Path to the JSON file storing mappings (relative to repo root or absolute).
            llm_model: LLM model to use for resolution.
        """
        self.mapping_file = Path(mapping_file)
        self.llm_model = llm_model
        self.mappings: Dict[str, str] = {}
        self._load_mappings()

    def _load_mappings(self) -> None:
        """Load mappings from JSON file."""
        if self.mapping_file.exists():
            try:
                with open(self.mapping_file, "r") as f:
                    self.mappings = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load acronym mappings: {e}")
                self.mappings = {}
        else:
            self.mappings = {}

    def _save_mappings(self) -> None:
        """Save mappings to JSON file."""
        try:
            with open(self.mapping_file, "w") as f:
                json.dump(self.mappings, f, indent=2, sort_keys=True)
        except Exception as e:
            print(f"Warning: Failed to save acronym mappings: {e}")

    def resolve(self, term: str, context: str = "") -> str:
        """
        Resolve a term to its full form if it's an acronym.
        
        1. Check local cache.
        2. If not found and looks like an acronym (all caps, short), ask LLM.
        3. If resolved, update cache and return full form.
        4. If not resolved or not an acronym, return original term.
        
        Args:
            term: The term to resolve (e.g., "FSR", "MZI").
            context: Context string to help the LLM (optional).
            
        Returns:
            The resolved full name (e.g., "Free Spectral Range", "Mach-Zehnder Interferometer"),
            or the original term if no resolution found.
        """
        clean_term = term.strip()
        
        # 1. Check direct cache hit
        if clean_term in self.mappings:
            return self.mappings[clean_term]
        
        # 2. Heuristic: Is it likely an acronym?
        # Criteria: mostly uppercase, length < 8, no spaces (usually)
        # We allow some tolerance (e.g. "Q-factor" might be borderline, but "MZM" is clear)
        is_likely_acronym = (
            clean_term.isupper() 
            and len(clean_term) < 8 
            and " " not in clean_term
        )
        
        if not is_likely_acronym:
            return clean_term
        
        # 3. Ask LLM for resolution
        resolved = self._ask_llm_for_resolution(clean_term, context)
        
        if resolved and resolved != clean_term:
            # Update cache
            self.mappings[clean_term] = resolved
            self._save_mappings()
            return resolved
            
        return clean_term

    def expand_acronyms_in_phrase(self, phrase: str, context: str = "") -> Tuple[str, Dict[str, str]]:
        """
        Expand acronym tokens inside a multi-word phrase and persist mappings.

        Example:
          "quad TIA" -> "quad transimpedance amplifier"

        Returns:
          (expanded_phrase, new_mappings_added)
        """
        original = (phrase or "").strip()
        if not original:
            return original, {}

        new_mappings: Dict[str, str] = {}

        # If the phrase contains lowercase letters, we prefer lowercasing expansions to fit sentence-case.
        phrase_is_mixed_or_lower = any(c.islower() for c in original)

        def _resolve_and_track(acronym: str) -> Optional[str]:
            """Resolve an acronym (normalized) and track newly added mappings."""
            key = (acronym or "").strip().upper()
            if not key:
                return None
            existed = key in self.mappings
            resolved = self.resolve(key, context=context)
            if not resolved or resolved == key:
                return None
            if not existed and key in self.mappings:
                new_mappings[key] = self.mappings[key]
            return resolved.lower() if phrase_is_mixed_or_lower else resolved

        expanded = original

        # Pass 1: plural/possessive acronym forms inside phrases, e.g.:
        # - "MZMs" -> "Mach-Zehnder Modulator"
        # - "MMIs" -> "Multimode Interferometer"
        # - "MZM's" -> "Mach-Zehnder Modulator"
        #
        # Note: We intentionally replace with the singular expanded form to canonicalize entity names.
        plural_or_possessive = re.compile(r"\b([A-Z]{2,8})(?:'s|s|es)\b")

        def _plural_cb(m: re.Match) -> str:
            base = m.group(1)
            rep = _resolve_and_track(base)
            return rep if rep is not None else m.group(0)

        expanded = plural_or_possessive.sub(_plural_cb, expanded)

        # Pass 2: exact acronym tokens (all-caps), e.g. "TIA"
        exact = re.compile(r"\b([A-Z]{2,8})\b")

        def _exact_cb(m: re.Match) -> str:
            tok = m.group(1)
            rep = _resolve_and_track(tok)
            return rep if rep is not None else tok

        expanded = exact.sub(_exact_cb, expanded)

        # Persist any newly learned mappings (resolve() already persists, but this keeps behavior consistent
        # if resolve() ever changes to delay writes).
        if new_mappings:
            self._save_mappings()

        return expanded, new_mappings

    def _ask_llm_for_resolution(self, term: str, context: str) -> Optional[str]:
        """Ask LLM to resolve acronym."""
        
        sys_prompt = """You are an assistant managing a Knowledge Base for Photonic Integrated Circuits.
Your goal is to expand standard photonics acronyms to their full names.

Examples:
- "FSR" -> "Free Spectral Range"
- "MZI" -> "Mach-Zehnder Interferometer"
- "SOI" -> "Silicon on Insulator"

If the term is NOT a known photonics acronym or ambiguous, return the original term.
Do NOT convert spaces to underscores. Keep standard English formatting.
"""

        user_prompt = f"""Expand the following acronym: "{term}"
Context: "{context}"

Return ONLY the expanded full name string. No markdown, no quotes, no extra text.
If you cannot expand it confidently, return "{term}"."""

        try:
            response = llm_api.call_llm(user_prompt, sys_prompt, self.llm_model)
            if response:
                cleaned = response.strip().strip('"').strip("'")
                # Basic validation: if it came back identical or super long, ignore
                if len(cleaned) > 100:
                    return None
                return cleaned
        except Exception as e:
            print(f"AcronymAgent LLM error: {e}")
            
        return None

