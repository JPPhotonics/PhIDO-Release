"""YAML file loader and parser for ontology files."""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class YAMLLoader:
    """Load and parse YAML ontology files."""
    
    def __init__(self, yaml_dir: Path):
        """Initialize YAML loader with directory path."""
        self.yaml_dir = Path(yaml_dir)
        if not self.yaml_dir.exists():
            raise ValueError(f"YAML directory does not exist: {yaml_dir}")
    
    def load_all_files(self) -> Dict[str, Any]:
        """Load all YAML files from the directory."""
        yaml_files = list(self.yaml_dir.glob("*.yaml"))
        yaml_files.extend(self.yaml_dir.glob("*.yml"))
        
        data = {
            "components": [],
            "architectures": [],
            "properties": [],
            "design_functions": [],
            "physical_principles": [],
        }
        
        for yaml_file in yaml_files:
            # Skip db_index.yaml as it's a reference file
            if yaml_file.name == "db_index.yaml":
                continue
            
            try:
                parsed = self.load_file(yaml_file)
                if parsed:
                    entity_type = parsed.get("entity_type")
                    if entity_type in data:
                        # If file contains multiple entities, extend the list
                        if "entities" in parsed:
                            data[entity_type].extend(parsed["entities"])
                        else:
                            data[entity_type].append(parsed)
            except Exception as e:
                print(f"Warning: Failed to load {yaml_file.name}: {e}")
        
        return data
    
    def load_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load and parse a single YAML file."""
        # Try different YAML loaders
        content = None
        
        # Check if this is a component file (starts with _) and use manual parser
        if file_path.stem.startswith("_") or file_path.name.startswith("_"):
            return self._parse_component_file_manual(file_path)
        
        # First try safe_load
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
        except yaml.YAMLError as e:
            # Try with FullLoader (more lenient)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError:
                # If both fail, check if it has Primitive: and use manual parsing
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_content = f.read()
                    if "Primitive:" in raw_content:
                        return self._parse_component_file_manual(file_path)
                # For properties/design_functions/physical_principles, try manual parsing
                if file_path.name in ["properties.yaml", "design_functions.yaml", "physical_principles.yaml"]:
                    # These files might have parsing issues, but let's try to handle them
                    print(f"Warning: YAML parsing failed for {file_path.name}, attempting manual parse")
                    return self._parse_entity_file_manual(file_path)
                return None
        
        if not content:
            return None
        
        # Handle different YAML structures
        if isinstance(content, dict):
            if "Primitive:" in content or any(key.startswith("_") for key in content.keys()):
                return self._parse_component_file(content, file_path)
            elif "LIST OF" in str(content):
                return None  # Skip index files
            else:
                # Check if it's a property, function, or principle
                return self._parse_entity_file(content, file_path)
        
        return None
    
    def _parse_component_file_manual(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Manually parse component files that have non-standard YAML format."""
        content = {}
        
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        current_key = None
        current_list = None
        
        for i, line in enumerate(lines):
            original_line = line
            line_stripped = line.rstrip()
            
            # Skip empty lines
            if not line_stripped.strip():
                continue
            
            # Check for Primitive: key
            if line_stripped.startswith("Primitive:"):
                primitive_name = line_stripped.split("Primitive:", 1)[1].strip()
                content["Primitive:"] = primitive_name
                # Save previous key before starting new section
                if current_key and current_list is not None:
                    content[current_key] = current_list
                current_key = None
                current_list = None
            # Check for top-level key (starts at column 0, has colon)
            # OR second-level key (starts with exactly 2 spaces, has colon, not a list item)
            elif (not line.startswith(" ") or (line.startswith("  ") and not line.startswith("  -"))) and ":" in line_stripped and not line_stripped.startswith("#"):
                # Save previous key's value before starting new key
                if current_key:
                    if current_list is not None:
                        content[current_key] = current_list
                    elif current_key not in content:
                        # Key was set but no value was captured, set to empty list
                        content[current_key] = []
                
                # Parse new key (strip leading spaces for key name)
                parts = line_stripped.split(":", 1)
                current_key = parts[0].strip()
                value_part = parts[1].strip() if len(parts) > 1 else ""
                
                # Check if next non-empty line starts a list (indented with 4 spaces and "-")
                # Lists under keys are indented with 4 spaces (2 for key level + 2 for list)
                next_is_list = False
                for j in range(i + 1, min(i + 5, len(lines))):
                    next_line = lines[j]
                    if not next_line.strip():
                        continue
                    # Check if line is indented with 4 spaces and starts with "-"
                    if next_line.startswith("    -"):
                        next_is_list = True
                        break
                    # If we hit a line at same or less indentation, it's not a list
                    if not next_line.startswith("    "):
                        break
                
                if next_is_list:
                    current_list = []
                    if value_part:
                        current_list.append(value_part)
                else:
                    content[current_key] = value_part
                    current_list = None
            # Check for list item (indented with 4 spaces, then "- ")
            # Lists under keys are indented with 4 spaces (2 for key level + 2 for list)
            elif line.startswith("    -") and current_key:
                if current_list is None:
                    current_list = []
                # Remove "    - " prefix (4 spaces + dash + space)
                item = line[6:].strip() if len(line) > 6 else line[5:].strip()
                current_list.append(item)
            # Check for continuation of value (indented, not a list item)
            elif line.startswith("  ") and current_key and not line.startswith("  -"):
                value_part = line_stripped.strip()
                if current_list is not None:
                    # Append to last list item or add as new item
                    if current_list:
                        current_list[-1] += " " + value_part
                    else:
                        current_list.append(value_part)
                else:
                    # Append to string value
                    if current_key in content:
                        content[current_key] += " " + value_part
                    else:
                        content[current_key] = value_part
        
        # Save last key's value
        if current_key:
            if current_list is not None:
                content[current_key] = current_list
            elif current_key not in content:
                # Key was set but no value was captured
                content[current_key] = []
        
        # Debug: print what was parsed
        if "PERFORMS_FUNCTION" in content or "HAS_PROPERTY" in content or "BASED_ON_PRINCIPLE" in content:
            print(f"Debug: Parsed {file_path.name} - Keys: {list(content.keys())}")
            if "PERFORMS_FUNCTION" in content:
                print(f"  PERFORMS_FUNCTION: {content['PERFORMS_FUNCTION']}")
            if "HAS_PROPERTY" in content:
                print(f"  HAS_PROPERTY: {content['HAS_PROPERTY']}")
            if "BASED_ON_PRINCIPLE" in content:
                print(f"  BASED_ON_PRINCIPLE: {content['BASED_ON_PRINCIPLE']}")
        
        # Now parse as if it was loaded normally
        return self._parse_component_file(content, file_path)
    
    def _parse_component_file(self, content: Dict, file_path: Path) -> Dict[str, Any]:
        """Parse a component/architecture YAML file."""
        # Extract primitive name - check for "Primitive:" key
        primitive_name = None
        
        # Check if "Primitive:" is a direct key
        if "Primitive:" in content:
            primitive_name = content["Primitive:"]
            if isinstance(primitive_name, str):
                primitive_name = primitive_name.strip().replace("_", " ")
        else:
            # Check nested structure
            for key, value in content.items():
                if isinstance(value, dict) and "Primitive:" in value:
                    primitive_name = value.get("Primitive:", "").strip().replace("_", " ")
                    content = value
                    break
        
        if not primitive_name:
            # Try to get from first key
            first_key = list(content.keys())[0] if content else None
            if first_key and first_key != "Primitive:":
                primitive_name = first_key.replace("_", " ").title()
        
        if not primitive_name:
            primitive_name = file_path.stem.replace("_", " ").title()
        
        # Determine if it's a component or architecture
        # Only treat as architecture if Type explicitly contains "Architecture"
        # Files starting with "_" are not automatically architectures
        entity_type = content.get("Type", "")
        if isinstance(entity_type, str):
            entity_type_lower = entity_type.lower()
            # Check if Type contains "Architecture" (case-insensitive)
            is_architecture = "architecture" in entity_type_lower
        else:
            is_architecture = False
        
        # Extract relationships
        relationships = {
            "performs_function": [x.replace("_", " ") for x in content.get("PERFORMS_FUNCTION", [])],
            "based_on_principle": [x.replace("_", " ") for x in content.get("BASED_ON_PRINCIPLE", [])],
            "has_property": [x.replace("_", " ") for x in content.get("HAS_PROPERTY", [])],
            "uses_component": [x.replace("_", " ") for x in content.get("USES_COMPONENT", [])],
        }
        
        # Build description from source
        description = content.get("Description", "")
        if not description:
            description = f"Photonic {primitive_name}"
        
        entity_data = {
            "name": primitive_name,
            "type": content.get("Type", "Component"),
            "source": content.get("Source", ""),
            "description": description,
            "relationships": relationships,
            "file_path": str(file_path),
        }
        
        if is_architecture:
            entity_data["entity_type"] = "architectures"
        else:
            entity_data["entity_type"] = "components"
        
        return entity_data
    
    def _parse_entity_file(self, content: Dict, file_path: Path) -> Dict[str, Any]:
        """Parse a property, design function, or physical principle file."""
        # Properties file has multiple entries
        if file_path.name == "properties.yaml":
            # This file contains multiple properties
            entities = []
            for prop_name, prop_data in content.items():
                if isinstance(prop_data, dict):
                    # Replace underscores with spaces in name
                    clean_name = prop_name.replace("_", " ")
                    entities.append(self._parse_property(clean_name, prop_data, file_path))
            return {"entity_type": "properties", "entities": entities}
        
        # Design functions file
        elif file_path.name == "design_functions.yaml":
            entities = []
            for func_name, func_data in content.items():
                if isinstance(func_data, dict):
                    # Replace underscores with spaces in name
                    clean_name = func_name.replace("_", " ")
                    entities.append(self._parse_design_function(clean_name, func_data, file_path))
            return {"entity_type": "design_functions", "entities": entities}
        
        # Physical principles file
        elif file_path.name == "physical_principles.yaml":
            entities = []
            for principle_name, principle_data in content.items():
                if isinstance(principle_data, dict):
                    # Replace underscores with spaces in name
                    clean_name = principle_name.replace("_", " ")
                    entities.append(self._parse_physical_principle(clean_name, principle_data, file_path))
            return {"entity_type": "physical_principles", "entities": entities}
        
        return None
    
    def _parse_property(self, name: str, data: Dict, file_path: Path) -> Dict[str, Any]:
        """Parse a single property entry."""
        return {
            "name": name,
            "source": data.get("Source", ""),
            "description": data.get("Description", ""),
            "units": data.get("Units", ""),
            "equations": data.get("Equations", []),
            "file_path": str(file_path),
        }
    
    def _parse_design_function(self, name: str, data: Dict, file_path: Path) -> Dict[str, Any]:
        """Parse a single design function entry."""
        return {
            "name": name,
            "source": data.get("Source", ""),
            "description": data.get("Description", ""),
            "equations": data.get("Equations", []),
            "file_path": str(file_path),
        }
    
    def _parse_physical_principle(self, name: str, data: Dict, file_path: Path) -> Dict[str, Any]:
        """Parse a single physical principle entry."""
        return {
            "name": name,
            "source": data.get("Source", ""),
            "description": data.get("Description", ""),
            "equations": data.get("Equations", []),
            "file_path": str(file_path),
        }
    
    def _parse_entity_file_manual(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Manually parse entity files (properties, design_functions, physical_principles)."""
        entities = []
        current_entity_name = None
        current_entity_data = {}
        current_key = None
        current_list = None
        
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            line = line.rstrip()
            
            # Skip empty lines and comments
            if not line.strip() or line.strip().startswith("#"):
                continue
            
            # Check if this is a new entity (top-level key, no indentation, ends with colon)
            if not line.startswith(" ") and line.endswith(":") and ":" in line:
                # Save previous entity
                if current_entity_name and current_entity_data:
                    entities.append(self._create_entity_dict(file_path, current_entity_name, current_entity_data))
                
                # Start new entity
                current_entity_name = line[:-1].strip().replace("_", " ")  # Remove colon and replace underscores
                current_entity_data = {}
                current_key = None
                current_list = None
            # Check for key-value pair (indented with 2 spaces)
            elif line.startswith("  ") and ":" in line and not line.strip().startswith("-"):
                # Save previous list if exists
                if current_key and current_list is not None:
                    current_entity_data[current_key] = current_list
                    current_list = None
                
                parts = line.split(":", 1)
                current_key = parts[0].strip()
                value_part = parts[1].strip() if len(parts) > 1 else ""
                
                # Check if next line starts a list
                if i + 1 < len(lines) and lines[i + 1].strip().startswith("-"):
                    current_list = []
                    if value_part:
                        current_list.append(value_part)
                else:
                    current_entity_data[current_key] = value_part
            # Check for list item (indented with 2 spaces, then "- ")
            elif line.startswith("  -") and current_key:
                if current_list is None:
                    current_list = []
                item = line.strip()[2:].strip()  # Remove "- " prefix
                current_list.append(item)
            # Check for continuation of description or other text (indented, not a list)
            elif line.startswith("  ") and current_key and not line.strip().startswith("-"):
                value_part = line.strip()
                if current_list is not None:
                    if current_list:
                        current_list[-1] += " " + value_part
                    else:
                        current_list.append(value_part)
                else:
                    # Append to string value
                    if current_key in current_entity_data:
                        current_entity_data[current_key] += " " + value_part
                    else:
                        current_entity_data[current_key] = value_part
        
        # Save last entity
        if current_entity_name and current_entity_data:
            if current_key and current_list is not None:
                current_entity_data[current_key] = current_list
            entities.append(self._create_entity_dict(file_path, current_entity_name, current_entity_data))
        
        # Determine entity type based on filename
        if file_path.name == "properties.yaml":
            return {"entity_type": "properties", "entities": entities}
        elif file_path.name == "design_functions.yaml":
            return {"entity_type": "design_functions", "entities": entities}
        elif file_path.name == "physical_principles.yaml":
            return {"entity_type": "physical_principles", "entities": entities}
        
        return None
    
    def _create_entity_dict(self, file_path: Path, name: str, data: Dict) -> Dict[str, Any]:
        """Create entity dictionary based on file type."""
        if file_path.name == "properties.yaml":
            return {
                "name": name,
                "source": data.get("Source", ""),
                "description": data.get("Description", ""),
                "units": data.get("Units", ""),
                "equations": data.get("Equations", []),
                "file_path": str(file_path),
            }
        elif file_path.name in ["design_functions.yaml", "physical_principles.yaml"]:
            return {
                "name": name,
                "source": data.get("Source", ""),
                "description": data.get("Description", ""),
                "equations": data.get("Equations", []),
                "file_path": str(file_path),
            }
        return {}

