"""Copy the numbers the website renders from results/ into site/data/inv03.json. Never edited by hand.

    python export_site_data.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
SITE = HERE.parents[1] / "site" / "data"
R = HERE / "results"
a = json.loads((R / "analysis.json").read_text())
m = json.loads((R / "milestone1.json").read_text())
c = json.loads((R / "counters.json").read_text())
out = {"info": m["info"], "analysis": a, "counters": c,
       "chase": [{k: r[k] for k in ("bytes", "cycles", "ns")} for r in m["chase"]]}
extra = R / "milestone2.json"
if extra.exists():
    out["m2"] = json.loads(extra.read_text())
SITE.mkdir(parents=True, exist_ok=True)
(SITE / "inv03.json").write_text(json.dumps(out, separators=(",", ":")))
print("wrote", SITE / "inv03.json", round((SITE / "inv03.json").stat().st_size / 1024, 1), "KB")
