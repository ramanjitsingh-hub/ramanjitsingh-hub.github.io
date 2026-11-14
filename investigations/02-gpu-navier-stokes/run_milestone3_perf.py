"""Milestone 3b: effect of the temporally blocked smoother. Writes results/milestone3_perf.json."""
import json
from run_experiments import call, OUT

res = {"vcycle_ms": {}, "bench": {}}
for prec in ("f64", "mixed"):
    for tb in (0, 1):  # tb applies to levels with N >= 512
        res["vcycle_ms"][f"{prec}_tb{tb}"] = {n: call("mg", "--N", n, "--prec", prec, "--tb", tb)["ms_per_cycle"] for n in (128, 256, 512, 1024, 2048)}
        res["bench"][f"{prec}_tb{tb}"] = {n: call("bench", "--N", n, "--prec", prec, "--tb", tb, "--steps", 20 if n < 2048 else 8)["ms_per_step_median"]
                                         for n in (128, 256, 512, 1024, 2048)}
        print(prec, tb, res["vcycle_ms"][f"{prec}_tb{tb}"], res["bench"][f"{prec}_tb{tb}"], flush=True)
(OUT / "milestone3_perf.json").write_text(json.dumps(res, indent=1))
