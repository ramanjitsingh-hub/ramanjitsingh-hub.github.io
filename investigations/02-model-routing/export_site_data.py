"""Copy the published numbers into site/data/inv02.json. Never edit that file by hand."""
import json
from pathlib import Path

HERE = Path(__file__).parent
SITE = HERE.parent.parent / "site" / "data" / "inv02.json"


def thin(curve, n=120):
    """Keep at most n points of a hull, always keeping both ends."""
    if len(curve) <= n:
        return curve
    step = (len(curve) - 1) / (n - 1)
    return [curve[round(i * step)] for i in range(n)]


def main():
    r = json.loads((HERE / "results" / "milestone1.json").read_text())
    r["oracle"]["hull"] = thin(r["oracle"]["hull"])
    for v in r["routers"].values():
        v["hull"] = thin(v["hull"])
    for v in r["lofo"].values():
        for k in ("hull", "single_hull", "fixed_ranking_hull"):
            if k in v:
                v[k] = thin(v[k], 60)
    m2 = HERE / "results" / "milestone2.json"
    if m2.exists():
        r["m2"] = json.loads(m2.read_text())
    SITE.write_text(json.dumps(r, separators=(",", ":")))
    print(f"wrote {SITE} ({SITE.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
