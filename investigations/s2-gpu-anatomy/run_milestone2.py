"""Milestone 2: experiments designed to decide between explanations of the three gaps left by milestone 1.
Writes results/milestone2.json. Close other GPU users first.

  fp64    warps per SM x {1, 8} independent chains: where the FP64 floor of 19 cycles appears, and
          whether one warp can reach it. Plus Nsight stall reasons at the floor.
  bw2     one warp per SM, K = 1..16, plain against software-pipelined loop, DRAM against an
          L2-resident array: is the single-warp cap a concurrency limit (scales with latency) or a rate limit?
  tlb     pointer chase touching one 128-byte line per page-sized block, few enough lines to stay in L2,
          for 4 KB, 64 KB and 2 MB blocks: a translation-reach step shows up as a latency step at a
          fixed number of blocks for the true page size.
  stride2 strides around 64 at a fixed 128 MB span, and every other warp shifted by 128 bytes:
          aliasing predicts dips only at powers of two that the shift removes.

    python run_milestone2.py [fp64 bw2 tlb tlb2 stride2 counters]
"""
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
EXE = str(HERE / "bin" / ("anat.exe" if sys.platform == "win32" else "anat"))
NCU = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.4.0\target\windows-desktop-win7-x64\ncu.exe"
OUT = HERE / "results" / "milestone2.json"


def call(*args):
    out = subprocess.run([EXE, *map(str, args)], capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def fp64():
    rows = []
    for c in (1, 8):
        for w in (1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32):
            r = call("fp64", "--warps", w, "--chains", c); rows.append(r); print("fp64", w, c, r["cycles_per_warp_fma_per_sm"], flush=True)
    return rows


def bw2():
    rows = []
    for arr, passes in ((256 << 20, 1), (512 << 10, 512)):
        for pipe in (0, 1):
            for k in (1, 2, 4, 8, 16):
                # the small array is read with L1 bypassed, otherwise each SM's 26 KB share sits in its L1
                r = call("bw", "--warps", 1, "--ilp", k, "--array", arr, "--passes", passes, "--pipe", pipe, "--cg", 1 if passes > 1 else 0)
                rows.append(r); print("bw2", arr, pipe, k, r["GBps"], flush=True)
    for w in (2, 4):
        for pipe in (0, 1):
            r = call("bw", "--warps", w, "--ilp", 1, "--pipe", pipe); rows.append(r); print("bw2", w, pipe, r["GBps"], flush=True)
    return rows


def tlb():
    rows = []
    for g, pmax in ((4096, 4096), (65536, 4096), (2 << 20, 512)):
        p = 4
        while p <= pmax:
            r = call("chase", "--bytes", p * g, "--stride", g, "--jitter", 1, "--cg", 1); r["blocks"] = p; rows.append(r)   # L2 only
            print("tlb", g, p, r["cycles"], flush=True); p *= 2
    return rows


def stride2():
    rows = []
    span = 128 << 20
    for s in (8, 16, 32, 48, 56, 60, 63, 64, 65, 72, 96, 128, 192, 256):
        r = call("stride", "--s", s, "--span", span); rows.append(r); print("stride2", s, r["useful_GBps"], flush=True)
    for s in (48, 64, 128, 256):
        r = call("stride", "--s", s, "--span", span, "--halfoff", 32); rows.append(r); print("stride2 halfoff", s, r["useful_GBps"], flush=True)
    return rows


STALLS = ["math_pipe_throttle", "dispatch_stall", "not_selected", "selected", "wait", "short_scoreboard", "long_scoreboard",
          "lg_throttle", "no_instruction", "branch_resolving"]


def ncu(kernel, args, metrics):
    cmd = [NCU, "--kernel-name", f"regex:^{kernel}", "--launch-skip", "1", "--launch-count", "1", "--metrics", ",".join(metrics),
           "--csv", "--page", "raw", EXE, *map(str, args), "--reps", "1"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600).stdout
    rows = [r for r in csv.DictReader(io.StringIO("\n".join(l for l in out.splitlines() if l.startswith('"'))))]
    def num(x):
        try:
            return float(str(x).replace(",", ""))
        except ValueError:
            return None
    rows = [r for r in rows if num(r.get(metrics[0])) is not None]
    return {m: num(rows[0][m]) for m in metrics}


def counters():
    st = [f"smsp__warp_issue_stalled_{s}_per_warp_active.pct" for s in STALLS]
    res = {"fp64": [], "bw": []}
    for w, c in ((1, 8), (4, 8), (12, 1)):
        m = ncu("k_fp64", ["fp64", "--warps", w, "--chains", c],
                ["sm__inst_executed_pipe_fp64.avg", "sm__pipe_fp64_cycles_active.avg", "sm__cycles_elapsed.avg", "smsp__issue_active.avg.pct_of_peak_sustained_active"] + st)
        m.update({"warps": w, "chains": c}); res["fp64"].append(m); print("counters fp64", w, c, flush=True)
    for k in (1, 8):
        m = ncu("k_read", ["bw", "--warps", 1, "--ilp", k], ["l1tex__m_xbar2l1tex_read_sectors.sum", "gpu__time_duration.sum"] + st)
        m.update({"warps": 1, "ilp": k}); res["bw"].append(m); print("counters bw", k, flush=True)
    return res


def tlb2():
    """Larger blocks, then one block at a time around the step, to separate entry count from page size."""
    rows = []
    for g in (4 << 20, 8 << 20):
        for p in (4, 8, 16, 24, 32, 48, 64):
            r = call("chase", "--bytes", p * g, "--stride", g, "--jitter", 1, "--cg", 1); r["blocks"] = p; rows.append(r); print("tlb2", g, p, r["cycles"], flush=True)
    for g in (2 << 20, 8 << 20):
        for p in range(12, 25):
            r = call("chase", "--bytes", p * g, "--stride", g, "--jitter", 1, "--cg", 1); r["blocks"] = p; r["fine"] = True; rows.append(r); print("tlb2 fine", g, p, r["cycles"], flush=True)
    return rows


EXPS = {"fp64": fp64, "bw2": bw2, "tlb": tlb, "tlb2": tlb2, "stride2": stride2, "counters": counters}

if __name__ == "__main__":
    pick = sys.argv[1:] or list(EXPS)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in pick:
        res[name] = EXPS[name]()
        OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)
