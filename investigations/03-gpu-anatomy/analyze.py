"""Compare every milestone 1 measurement with the model stated for it. Writes results/analysis.json.

    python analyze.py
"""
import json
from math import gcd
from pathlib import Path
from statistics import median

HERE = Path(__file__).parent
R = json.loads((HERE / "results" / "milestone1.json").read_text())
info = R["info"]
SMS = info["sms"]
out = {}

# 1. latency staircase: plateaus by working-set size
ch = R["chase"]
def plateau(lo, hi):
    v = [r for r in ch if lo <= r["bytes"] <= hi]
    return {"cycles": round(median(r["cycles"] for r in v), 1), "ns": round(median(r["ns"] for r in v), 1), "from_bytes": lo, "to_bytes": hi, "points": len(v)}
out["latency"] = {"L1": plateau(4096, 92672), "L2": plateau(185344, 741376), "DRAM": plateau(1482880, 33554432),
                  "DRAM_large": plateau(67108864, 536870912),
                  "transitions_bytes": {"L1_end": [92672, 131072], "L2_end": [741376, 1482880]}, "l2_bytes_reported": info["l2_bytes"]}

# 2. Little's law: BW = min(Bpeak, bytes in flight / DRAM latency)
lat_ns = out["latency"]["DRAM"]["ns"]
bpeak = max(r["GBps"] for r in R["bw"])
bw_rows = []
for r in R["bw"]:
    inflight = SMS * r["warps_per_sm"] * r["ilp"] * 32 * 16
    pred = min(bpeak, inflight / lat_ns)            # bytes per ns = GB/s
    bw_rows.append({"warps": r["warps_per_sm"], "ilp": r["ilp"], "inflight_bytes": inflight, "measured": r["GBps"], "predicted": round(pred, 2),
                    "ratio": round(r["GBps"] / pred, 3)})
latency_bound = [x for x in bw_rows if x["predicted"] < 0.8 * bpeak]
out["littles_law"] = {"dram_latency_ns": lat_ns, "peak_GBps": bpeak, "spec_GBps": info["mem_clock_khz"] * 1e3 * 2 * info["bus_bits"] / 8 / 1e9,
                      "rows": bw_rows, "latency_bound_ratio_range": [min(x["ratio"] for x in latency_bound), max(x["ratio"] for x in latency_bound)]}

# 3. coalescing: useful BW = Bpeak / min(8, S)
st = []
for r in R["stride"]:
    pred = bpeak / min(8, r["s"])
    st.append({"s": r["s"], "measured": r["useful_GBps"], "predicted": round(pred, 2), "ratio": round(r["useful_GBps"] / pred, 3)})
out["coalescing"] = {"rows": st}

# 4. roofline: P = min(peak at measured clock, AI x Bpeak)
ai = []
for r in R["ai"]:
    lanes = 128 if r["prec"] == "f32" else 2      # FP32 lanes and FP64 lanes per SM, Ampere GA10x
    peak = SMS * lanes * 2 * r["clock_GHz"]
    pred = min(peak, r["intensity"] * bpeak)
    ai.append({"prec": r["prec"], "F": r["flops_per_elem"], "intensity": r["intensity"], "clock_GHz": r["clock_GHz"], "measured": r["GFLOPs"],
               "peak": round(peak, 1), "predicted": round(pred, 1), "ratio": round(r["GFLOPs"] / pred, 3)})
out["roofline"] = {"rows": ai}

# 5. divergence: cycles = c1 x min(active, paths)
dv = R["div"]
c1 = next(r for r in dv if r["active"] == 32 and r["paths"] == 1)["cycles_per_warp_fma_per_sm"]
drows = []
for r in dv:
    k = min(r["active"], r["paths"])
    pred = c1 * k
    drows.append({"active": r["active"], "paths": r["paths"], "measured": r["cycles_per_warp_fma_per_sm"], "predicted": round(pred, 3),
                  "ratio": round(r["cycles_per_warp_fma_per_sm"] / pred, 4) if pred else None})
out["divergence"] = {"c1_cycles": c1, "pipe_model_cycles": 16, "rows": drows,
                     "max_rel_err": max(abs(x["ratio"] - 1) for x in drows if x["ratio"])}

# 6. shared-memory banks: cycles = max(2, wavefronts)
def wavefronts(width, s):
    words = width // 4
    lanes = [[(l * s * words) + q for q in range(words)] for l in range(32)]
    if s == 0:
        return 1                                   # one address for the whole warp: a single broadcast
    group = 128 // width                           # lanes served per 128-byte wavefront
    total = 0
    for g in range(0, 32, group):
        banks = {}
        for l in range(g, g + group):
            for w in lanes[l]:
                banks.setdefault(w % 32, set()).add(w)
        total += max(len(v) for v in banks.values())
    return total
bk = []
for r in R["bank"]:
    p = max(2, 1 if r["s"] == 0 else gcd(r["s"], 32))
    bk.append({"width": 4, "s": r["s"], "measured": r["cycles_per_warp_load_per_sm"], "predicted": p, "ratio": round(r["cycles_per_warp_load_per_sm"] / p, 3)})
for r in R.get("bankw", []):
    wf = wavefronts(r["width"], r["s"])
    p = max(2, wf)
    bk.append({"width": r["width"], "s": r["s"], "wavefronts": wf, "measured": r["cycles_per_warp_load_per_sm"], "predicted": p,
               "ratio": round(r["cycles_per_warp_load_per_sm"] / p, 3), "bytes_per_cycle": r["bytes_per_cycle_per_sm"]})
out["banks"] = {"rows": bk, "max_rel_err": max(abs(x["ratio"] - 1) for x in bk)}

(HERE / "results" / "analysis.json").write_text(json.dumps(out, indent=1))

# summary
L = out["latency"]
print(f"latency  L1 {L['L1']['cycles']} cyc  L2 {L['L2']['cycles']} cyc  DRAM {L['DRAM']['cycles']} cyc ({L['DRAM']['ns']} ns)  >64MB {L['DRAM_large']['cycles']} cyc")
ll = out["littles_law"]
print(f"little   peak {ll['peak_GBps']} of spec {ll['spec_GBps']:.1f} GB/s; latency-bound points measured/predicted {ll['latency_bound_ratio_range']}")
for x in ll["rows"]:
    if x["warps"] == 1 or x["ilp"] == 1 and x["warps"] <= 4:
        print("         ", x)
print("coalesce", [(x["s"], x["ratio"]) for x in out["coalescing"]["rows"]])
print("roofline", [(x["prec"], x["F"], x["ratio"]) for x in out["roofline"]["rows"]])
print(f"diverge  c1 {c1} cycles, max rel err {out['divergence']['max_rel_err']:.4f}")
print(f"banks    max rel err {out['banks']['max_rel_err']:.3f}")
print("          worst", sorted(out["banks"]["rows"], key=lambda x: -abs(x["ratio"] - 1))[:4])

# ---------------------------------------------------------------- milestone 2
M2P = HERE / "results" / "milestone2.json"
if M2P.exists():
    M2 = json.loads(M2P.read_text())
    m2 = {}
    f = M2["fp64"]
    dep1 = next(r for r in f if r["chains"] == 1 and r["warps_per_sm"] == 1)["cycles_per_warp_fma_per_sm"]
    floor = min(r["cycles_per_warp_fma_per_sm"] for r in f)
    sat = {c: min(r["warps_per_sm"] for r in f if r["chains"] == c and r["cycles_per_warp_fma_per_sm"] < floor * 1.01) for c in (1, 8)}
    cf = M2["counters"]["fp64"]
    pipe_per_inst = [r["sm__pipe_fp64_cycles_active.avg"] / r["sm__inst_executed_pipe_fp64.avg"] for r in cf]
    m2["fp64"] = {"rows": [{k: r[k] for k in ("warps_per_sm", "chains", "cycles_per_warp_fma_per_sm")} for r in f],
                  "dependent_latency_cycles": dep1, "floor_cycles": floor, "saturating_warps": sat, "littles_warps": round(dep1 / floor, 2),
                  "single_warp_8_chains": next(r for r in f if r["chains"] == 8 and r["warps_per_sm"] == 1)["cycles_per_warp_fma_per_sm"],
                  "pipe_active_per_inst": [round(x, 2) for x in pipe_per_inst],
                  "stalls": [{"warps": r["warps"], "chains": r["chains"],
                              **{s: round(r[f"smsp__warp_issue_stalled_{s}_per_warp_active.pct"], 1) for s in ("short_scoreboard", "math_pipe_throttle", "wait", "selected")}} for r in cf]}
    b = M2["bw2"]
    def g(arr, pipe, k): return next(r["GBps"] for r in b if r["warps_per_sm"] == 1 and r["array"] == arr and r["pipe"] == pipe and r["ilp"] == k)
    l2_ns = out["latency"]["L2"]["ns"]
    m2["bw"] = {"dram": {p: {k: g(256 << 20, p, k) for k in (1, 2, 4, 8, 16)} for p in (0, 1)},
                "l2": {p: {k: g(512 << 10, p, k) for k in (1, 2, 4, 8, 16)} for p in (0, 1)},
                "l2_latency_ns": l2_ns, "l2_littles_K1": round(SMS * 512 / l2_ns, 1),
                "dram_littles_K1": round(SMS * 512 / lat_ns, 1)}
    t = M2["tlb"] + M2.get("tlb2", [])
    steps = {}
    for gsz in sorted({r["stride"] for r in t}):
        rs = sorted([r for r in t if r["stride"] == gsz], key=lambda r: r["blocks"])
        base = median(r["cycles"] for r in rs[:3])
        flat = [r["blocks"] for r in rs if r["cycles"] < base + 4]
        up = [r["blocks"] for r in rs if r["cycles"] >= base + 4]
        after = [x for x in up if x > max(flat)]          # a raised point below the last flat one is an outlier, not the step
        steps[gsz] = {"base_cycles": round(base, 1), "last_flat_blocks": max(flat), "first_step_blocks": min(after) if after else None,
                      "outliers": [x for x in up if x < max(flat)],
                      "step_cycles": round(median(r["cycles"] for r in rs if r["blocks"] in after) - base, 1) if after else None,
                      "points": [[r["blocks"], r["cycles"]] for r in rs]}
    m2["tlb"] = steps
    m2["stride"] = [{k: r[k] for k in ("s", "halfoff", "useful_GBps")} for r in M2["stride2"]]
    out["m2"] = m2
    (HERE / "results" / "analysis.json").write_text(json.dumps(out, indent=1))
    print("m2 fp64", {k: m2["fp64"][k] for k in ("dependent_latency_cycles", "floor_cycles", "saturating_warps", "littles_warps", "single_warp_8_chains", "pipe_active_per_inst")})
    print("m2 bw dram", m2["bw"]["dram"], "\n   l2", m2["bw"]["l2"], m2["bw"]["l2_littles_K1"], m2["bw"]["dram_littles_K1"])
    for gsz, v in steps.items(): print("m2 tlb", gsz, {k: v[k] for k in ("base_cycles", "last_flat_blocks", "first_step_blocks", "step_cycles", "outliers")})
