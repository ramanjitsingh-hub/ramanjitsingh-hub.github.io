"""Investigation 02, milestone 2: latency ablation, mixed-precision multigrid, final benchmarks.

    ./build.sh && python run_milestone2.py      # writes results/milestone2.json
"""
import json
import math
import time
from pathlib import Path

from run_experiments import call, orders, OUT

ABLATION = [
    ("baseline (milestone 1)", ["--graph", 0, "--coarse1", 0, "--warm", 0, "--adaptive", 0]),
    ("+ single-launch coarse solve", ["--graph", 0, "--coarse1", 1, "--warm", 0, "--adaptive", 0]),
    ("+ CUDA Graph V-cycle", ["--graph", 1, "--coarse1", 1, "--warm", 0, "--adaptive", 0]),
    ("+ blind cycles, one check", ["--graph", 1, "--coarse1", 1, "--warm", 0, "--adaptive", 1]),
    ("+ warm start", ["--graph", 1, "--coarse1", 1, "--warm", 1, "--adaptive", 1]),
]


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    res = {"bandwidth": call("bw")}

    # E5  ablation, FP64, ms per step
    abl = {}
    for n in (128, 512, 2048):
        rows = []
        for name, flags in ABLATION:
            r = call("bench", "--N", n, "--steps", 20 if n < 2048 else 8, *flags)
            rows.append({"name": name, **{k: r[k] for k in ("ms_per_step_median", "ms_per_step_min", "ms_per_step_max",
                                                              "mg_cycles_per_solve", "host_syncs_per_step", "phase_ms_per_step")}})
        r = call("bench", "--N", n, "--steps", 20 if n < 2048 else 8, "--prec", "mixed")
        rows.append({"name": "+ mixed-precision multigrid", **{k: r[k] for k in ("ms_per_step_median", "ms_per_step_min", "ms_per_step_max",
                                                                              "mg_cycles_per_solve", "host_syncs_per_step", "phase_ms_per_step")}})
        abl[n] = rows
        print("ablation", n, [(x["name"], x["ms_per_step_median"]) for x in rows], f"t={time.time()-t0:.0f}s")
    res["ablation"] = abl

    # E6  cost of one V-cycle, with and without graph, by precision
    cyc = {}
    for prec in ("f64", "f32", "mixed"):
        cyc[prec] = {}
        for g in (0, 1):
            cyc[prec][f"graph{g}"] = [call("mg", "--N", n, "--prec", prec, "--graph", g, "--coarse1", 1)["ms_per_cycle"]
                                      for n in (64, 128, 256, 512, 1024, 2048)]
    cyc["N"] = [64, 128, 256, 512, 1024, 2048]
    res["vcycle_ms"] = cyc
    print("vcycle", cyc, f"t={time.time()-t0:.0f}s")

    # E7  residual histories: FP64, FP32, mixed at N = 1024
    res["residual_history_N1024"] = {p: call("mg", "--N", 1024, "--prec", p)["rel_resid"] for p in ("f64", "f32", "mixed")}

    # E8  Taylor-Green convergence, mixed precision vs FP64 (optimised solver)
    Ns = [32, 64, 128, 256, 512, 1024]
    tg = {}
    for prec in ("f64", "mixed"):
        rows = [call("tg", "--N", n, "--prec", prec) for n in Ns]
        tg[prec] = {"N": Ns, "linf": [r["linf"] for r in rows], "order_linf": orders(Ns, [r["linf"] for r in rows]),
                    "wall_ms": [r["wall_ms"] for r in rows], "max_div": [r["max_div"] for r in rows]}
        print("tg", prec, tg[prec]["linf"], tg[prec]["wall_ms"], f"t={time.time()-t0:.0f}s")
    res["taylor_green"] = tg

    # E9  final throughput
    fin = {}
    for prec in ("f64", "mixed", "f32"):
        fin[prec] = [call("bench", "--N", n, "--prec", prec, "--steps", 20 if n < 2048 else 8) for n in (128, 256, 512, 1024, 2048)]
    res["final"] = fin
    res["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "milestone2.json").write_text(json.dumps(res, indent=1))
    print("done", res["runtime_s"])


if __name__ == "__main__":
    main()
