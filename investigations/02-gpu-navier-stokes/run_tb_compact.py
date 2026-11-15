"""Milestone 4a: colour-compacted thread mapping in the temporally blocked smoother.

The first k_smooth_tb skipped the other colour with a branch, so on every FP64 instruction half of
each warp was idle. --tbcompact 1 maps threads onto same-colour cells only. Arithmetic and update
order are unchanged, so results are bitwise identical (checked separately with --dump and cmp).
A/B is interleaved, 5 repeats each, median reported. Writes results/tb_compact.json.
    python run_tb_compact.py          # V-cycle A/B at N = 512, 1024, 2048
    python run_tb_compact.py --step   # full time step A/B at N = 2048, 3 interleaved repeats
"""
import json
import statistics
from run_experiments import call, OUT

import sys

NS = (512, 1024, 2048)
if "--step" in sys.argv:   # full time step A/B only, merged into the existing json
    res = json.loads((OUT / "tb_compact.json").read_text()); res["step_ms"] = {}
    for prec in ("f64", "mixed"):
        t = {0: [], 1: []}
        for _ in range(3):
            for c in (0, 1):
                t[c].append(call("bench", "--N", 2048, "--prec", prec, "--steps", 8, "--tbcompact", c)["ms_per_step_median"])
        m0, m1 = statistics.median(t[0]), statistics.median(t[1])
        res["step_ms"][f"{prec}_2048"] = {"branch": m0, "compact": m1, "speedup": round(m0 / m1, 3), "raw": t}
        print(prec, 2048, m0, m1, round(m0 / m1, 3), flush=True)
    (OUT / "tb_compact.json").write_text(json.dumps(res, indent=1))
    sys.exit()
res = {"repeats": 5, "vcycle_ms": {}}
for prec in ("f64", "mixed"):
    for n in NS:
        t = {0: [], 1: []}
        for _ in range(5):
            for c in (0, 1):
                t[c].append(call("mg", "--N", n, "--prec", prec, "--tbcompact", c)["ms_per_cycle"])
        m0, m1 = statistics.median(t[0]), statistics.median(t[1])
        res["vcycle_ms"][f"{prec}_{n}"] = {"branch": m0, "compact": m1, "speedup": round(m0 / m1, 3), "raw": t}
        print(prec, n, m0, m1, round(m0 / m1, 3), flush=True)
(OUT / "tb_compact.json").write_text(json.dumps(res, indent=1))
