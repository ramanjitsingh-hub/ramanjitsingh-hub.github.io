"""Roofline data for every kernel, from Nsight Compute hardware counters. Writes results/roofline.json.

Needs GPU performance counters enabled (NVIDIA Control Panel, Developer, Manage GPU Performance Counters).
Nsight Compute locks the SM clock to base frequency while profiling, so peaks are derived from the counters
themselves (achieved rate divided by the reported fraction of peak), not from datasheet numbers.

    python profile_roofline.py
"""
import csv
import io
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
EXE = str(HERE / "bin" / "ns2d.exe")
NCU = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.4.0\target\windows-desktop-win7-x64\ncu.exe"
N = 1024
OPS = ["dadd", "dmul", "dfma", "fadd", "fmul", "ffma"]
METRICS = ["gpu__time_duration.sum", "launch__grid_size", "dram__bytes_read.sum", "dram__bytes_write.sum",
           "sm__throughput.avg.pct_of_peak_sustained_elapsed", "dram__throughput.avg.pct_of_peak_sustained_elapsed",
           "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active", "sm__warps_active.avg.pct_of_peak_sustained_active",
           "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"] + \
          [f"sm__sass_thread_inst_executed_op_{o}_pred_on.sum" for o in OPS]

# (kernel regex, mode args, label)
JOBS = []
for prec in ("f64", "f32"):
    bench = ["bench", "--N", N, "--prec", prec, "--steps", 1]
    mg = ["mg", "--N", N, "--prec", prec]
    JOBS += [("k_rhs", bench, prec), ("k_stage", bench, prec), ("k_div", bench, prec), ("k_correct", bench, prec),
             ("k_smooth<", mg + ["--tb", 0], prec), ("k_smooth_tb", mg + ["--tbmin", 32], prec), ("k_resid", mg, prec),
             ("k_restrict", mg, prec), ("k_prolong_add", mg, prec), ("k_coarse", mg, prec)]


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return None


def profile(regex, args):
    cmd = [NCU, "--kernel-name", f"regex:{regex}", "--launch-skip", "30", "--launch-count", "12",
           "--metrics", ",".join(METRICS), "--csv", "--page", "raw", EXE, *map(str, args)]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800).stdout
    lines = [l for l in out.splitlines() if l.startswith('"')]
    rows = [r for r in csv.DictReader(io.StringIO("\n".join(lines))) if num(r.get("launch__grid_size")) is not None]
    if not rows:
        return None
    best = max(rows, key=lambda r: num(r["launch__grid_size"]))      # finest level
    same = [r for r in rows if r["launch__grid_size"] == best["launch__grid_size"]]
    avg = {m: sum(num(r[m]) for r in same) / len(same) for m in METRICS}
    t = avg["gpu__time_duration.sum"] * 1e-9
    byts = avg["dram__bytes_read.sum"] + avg["dram__bytes_write.sum"]
    fl64 = avg["sm__sass_thread_inst_executed_op_dadd_pred_on.sum"] + avg["sm__sass_thread_inst_executed_op_dmul_pred_on.sum"] + 2 * avg["sm__sass_thread_inst_executed_op_dfma_pred_on.sum"]
    fl32 = avg["sm__sass_thread_inst_executed_op_fadd_pred_on.sum"] + avg["sm__sass_thread_inst_executed_op_fmul_pred_on.sum"] + 2 * avg["sm__sass_thread_inst_executed_op_ffma_pred_on.sum"]
    return {"kernel": best["Kernel Name"].split("(")[0].replace("void ", ""), "grid": int(num(best["launch__grid_size"])), "samples": len(same),
            "time_us": round(t * 1e6, 3), "dram_bytes": byts, "flop64": fl64, "flop32": fl32,
            "GBps": round(byts / t / 1e9, 2), "GFLOPs64": round(fl64 / t / 1e9, 3), "GFLOPs32": round(fl32 / t / 1e9, 3),
            "intensity64": round(fl64 / byts, 4) if byts else None, "intensity32": round(fl32 / byts, 4) if byts else None,
            "sm_pct": round(avg["sm__throughput.avg.pct_of_peak_sustained_elapsed"], 1),
            "dram_pct": round(avg["dram__throughput.avg.pct_of_peak_sustained_elapsed"], 1),
            "fp64_pipe_pct": round(avg["sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active"], 1),
            "fma_pipe_pct": round(avg["sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"], 1),
            "occupancy_pct": round(avg["sm__warps_active.avg.pct_of_peak_sustained_active"], 1)}


def main():
    res = {"N": N, "kernels": []}
    for regex, args, prec in JOBS:
        r = profile(regex, args)
        if r:
            r["prec"] = prec; res["kernels"].append(r)
            print(f"{prec} {r['kernel']:28s} grid={r['grid']:6d} {r['time_us']:9.2f}us {r['GBps']:7.2f} GB/s dram {r['dram_pct']:5.1f}% "
                  f"sm {r['sm_pct']:5.1f}% fp64pipe {r['fp64_pipe_pct']:5.1f}% AI64 {r['intensity64']} AI32 {r['intensity32']}", flush=True)
        else:
            print("no data for", regex, prec, flush=True)
    # implied peaks at the profiling clock
    bw = [k["GBps"] / (k["dram_pct"] / 100) for k in res["kernels"] if k["dram_pct"] > 40]
    f64 = [k["GFLOPs64"] / (k["fp64_pipe_pct"] / 100) for k in res["kernels"] if k["fp64_pipe_pct"] > 40 and k["GFLOPs64"] > 0]
    res["implied_peak_GBps"] = round(sum(bw) / len(bw), 1) if bw else None
    res["implied_peak_GFLOPs64"] = round(sum(f64) / len(f64), 1) if f64 else None
    print("implied peaks at profiling clock:", res["implied_peak_GBps"], "GB/s,", res["implied_peak_GFLOPs64"], "FP64 GFLOP/s")
    (HERE / "results" / "roofline.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
