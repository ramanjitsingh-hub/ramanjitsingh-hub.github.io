"""Copy the published numbers into site/data/inv03.json. Never edit that file by hand."""
import json
from pathlib import Path

HERE = Path(__file__).parent
SITE = HERE.parent.parent / "site" / "data" / "inv03.json"
r = json.loads((HERE / "results" / "milestone1.json").read_text())
SITE.write_text(json.dumps(r, separators=(",", ":")))
print(f"wrote {SITE} ({SITE.stat().st_size // 1024} KB)")
