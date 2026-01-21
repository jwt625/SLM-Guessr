#!/usr/bin/env python3
"""
Update samples.json manifest without regenerating GIFs.
"""

import json
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent))
from slm_guessr.generator import get_all_samples

def main():
    """Update manifest with all sample configs."""
    samples = get_all_samples()
    
    manifest_entries = []
    for config in samples:
        entry = {
            "id": config.id,
            "level": config.level,
            "category": config.category,
            "name": config.name,
            "description": config.description,
            "phase_gif": f"assets/L{config.level}/{config.id}_phase.gif",
            "intensity_gif": f"assets/L{config.level}/{config.id}_intensity.gif",
            "parameters": config.parameters or {}
        }
        manifest_entries.append(entry)
    
    manifest = {
        "samples": manifest_entries,
        "generated_at": datetime.now().isoformat(),
        "version": "1.0.0",
    }
    
    # Save manifest
    manifest_path = Path(__file__).parent / "static" / "samples.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Updated manifest with {len(manifest_entries)} samples")
    print(f"L1 samples: {len([s for s in samples if s.level == 1])}")
    print(f"L2 samples: {len([s for s in samples if s.level == 2])}")
    print(f"Manifest saved to: {manifest_path}")

if __name__ == "__main__":
    main()

