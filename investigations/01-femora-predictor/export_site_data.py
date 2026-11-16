"""Copy the numbers the website needs from results/ into site/data/inv01.json.
The site never computes or edits results. It only renders this file.

    python export_site_data.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
SITE = HERE.parents[1] / "site" / "data"

s = json.loads((HERE / "results" / "summary.json").read_text())
f = json.loads((HERE / "results" / "figdata.json").read_text())

MODELS = ["production", "minimal_fix", "grid_bayes"]
KEYS = ["mae_days", "coverage_80", "mean_interval_width_days", "crps", "log_score",
        "coverage_by_cycle", "mae_by_cycle", "pit_hist"]

out = {"meta": s["meta"], "kf_vs_ewma": f["kf_vs_ewma"], "cohorts": {}}
for c in ["main/as_written", "main/as_intended", "clean/as_written", "misspecified/as_written"]:
    out["cohorts"][c] = {m: {k: s[c][m][k] for k in KEYS} for m in MODELS}
    out["cohorts"][c]["lockout"] = s[c]["lockout"]
    out["cohorts"][c]["brier"] = s[c]["daily_brier_final_cycle"]

ex = {}
for uid, e in f["examples"].items():
    tag = "shift" if e["meta"]["has_shift"] else ("long" if e["meta"]["mu0"] > 35 else "regular")
    ex[tag] = {k: e[k] for k in ["obs", "latent", "prod_mu", "prod_lo", "prod_hi", "gb_med", "gb_lo",
                                  "gb_hi", "accepted", "missed", "gb_level"]}
    ex[tag]["meta"] = {k: v for k, v in e["meta"].items() if k != "level_grid"}
    ex[tag]["level_grid"] = e["meta"]["level_grid"]
out["examples"] = ex
if (HERE / "results" / "conformal.json").exists():
    out["conformal"] = json.loads((HERE / "results" / "conformal.json").read_text())

SITE.mkdir(parents=True, exist_ok=True)
(SITE / "inv01.json").write_text(json.dumps(out, separators=(",", ":")))
print("wrote", SITE / "inv01.json", round((SITE / "inv01.json").stat().st_size / 1024, 1), "KB")
