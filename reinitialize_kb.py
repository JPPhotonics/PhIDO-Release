"""
Script to re-import YAML ontology data into ArangoDB.
This triggers the update process where underscores in entity names are replaced with spaces.
"""

import os
import sys
from pathlib import Path
from PhotonicsAI.KnowledgeBase.ArangoDB import KnowledgeBaseClient, ArangoDBConfig

# Set default password if not in env
if not os.getenv("ARANGO_PASSWORD"):
    # Try my_secure_password first as seen in container settings
    os.environ["ARANGO_PASSWORD"] = "my_secure_password"

def main():
    print("Initializing KB Client...")
    try:
        config = ArangoDBConfig(
            host=os.getenv("ARANGO_HOST", "localhost"),
            port=int(os.getenv("ARANGO_PORT", "8529")),
            username=os.getenv("ARANGO_USERNAME", "root"),
            password=os.getenv("ARANGO_PASSWORD", "my_password"),
            database=os.getenv("ARANGO_DATABASE", "photonics_kb")
        )
        client = KnowledgeBaseClient(config=config)
        client.connect()
        print("Connected to ArangoDB.")
    except Exception as e:
        print(f"Failed to connect to ArangoDB: {e}")
        return

    # Path to YAML files
    yaml_dir = Path("PhotonicsAI/KnowledgeBase/GenerativeOntology/Primitives")
    if not yaml_dir.exists():
        print(f"Error: YAML directory not found at {yaml_dir}")
        return

    print(f"Importing YAML data from {yaml_dir}...")
    try:
        # This calls the importer, which uses the updated YAMLLoader
        # YAMLLoader now replaces underscores with spaces in names
        # Importer sanitizes keys (underscores restored for _key) but updates 'name' field
        vertex_map = client.import_yaml_data(yaml_dir)
        print(f"Successfully imported {len(vertex_map)} entities.")
    except Exception as e:
        print(f"Error during import: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

