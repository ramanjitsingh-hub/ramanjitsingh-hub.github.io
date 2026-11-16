"""Milestone 1: six controlled experiments, each against a model written down before the run.
Writes results/milestone1.json. Close anything else using the GPU (for example a browser tab running
WebGPU) before running, because the experiments measure the whole device.

    python run_milestone1.py            # everything
    python run_milestone1.py chase bw   # selected experiments, merged into the existing json
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
EXE = str(HERE / "bin" / ("anat.exe" if sys.platform == "win32" else "anat"))
OUT = HERE / "results" / "milestone1.json"


def call(*args):
    out = subprocess.run([EXE, *map(str, args)], capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def chase():
    sizes = sorted({int(2 ** (k / 2)) // 128 * 128 for k in range(24, 59)})   # 4 KB .. 512 MB, half-octave steps
    rows = []
    for b in sizes:
        r = call("chase", "--bytes", b); rows.append(r); print("chase", b, r["cycles"], flush=True)
    return rows


def bw():
    rows = []
    for k in (1, 2, 4, 8):
        for w in (1, 2, 4, 8, 12, 16, 24, 32, 48):
            r = call("bw", "--warps", w, "--ilp", k); rows.append(r); print("bw", w, k, r["GBps"], flush=True)
    return rows


def stride():
    rows = []
    for s in list(range(1, 17)) + [24, 32, 48, 64]:
        r = call("stride", "--s", s); rows.append(r); print("stride", s, r["useful_GBps"], flush=True)
    return rows


def ai():
    rows = []
    for p in ("f32", "f64"):
        for f in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
            r = call("ai", "--prec", p, "--flops", f); rows.append(r); print("ai", p, f, r["GFLOPs"], r["clock_GHz"], flush=True)
    return rows


def div():
    combos = [(k, 1) for k in (0, 1, 2, 4, 8, 16, 32)] + [(32, p) for p in (2, 4, 8, 16, 32)] + [(8, 4), (4, 8), (16, 32), (2, 32)]
    rows = []
    for a, p in combos:
        r = call("div", "--active", a, "--paths", p); rows.append(r); print("div", a, p, r["cycles_per_warp_fma_per_sm"], flush=True)
    return rows


def bank():
    rows = []
    for s in range(0, 34):
        r = call("bank", "--s", s); rows.append(r); print("bank", s, r["cycles_per_warp_load_per_sm"], flush=True)
    return rows


def bankw():
    rows = []
    for w in (4, 8, 16):
        for s in list(range(0, 9)) + [16, 32]:
            r = call("bankw", "--width", w, "--s", s); rows.append(r); print("bankw", w, s, r["cycles_per_warp_load_per_sm"], flush=True)
    return rows


EXPS = {"chase": chase, "bw": bw, "stride": stride, "ai": ai, "div": div, "bank": bank, "bankw": bankw}

if __name__ == "__main__":
    pick = sys.argv[1:] or list(EXPS)
    res = json.loads(OUT.read_text()) if OUT.exists() and sys.argv[1:] else {}
    res["info"] = call("info")
    for name in pick:
        res[name] = EXPS[name]()
        OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)
