"""Systems study S1, milestone 1. Runs every experiment and writes results/summary.json.

    ./build.sh && python run_experiments.py
"""
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
EXE = str(HERE / "bin" / "ns2d.exe")
OUT = HERE / "results"


def call(*args):
    r = subprocess.run([EXE, *map(str, args)], capture_output=True, text=True, check=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


def orders(Ns, errs):
    return [round(math.log(errs[k] / errs[k + 1]) / math.log(Ns[k + 1] / Ns[k]), 3) for k in range(len(errs) - 1)]


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    res = {"meta": {"python": platform.python_version(), "numpy": np.__version__}}
    res["bandwidth"] = call("bw")
    print("bandwidth", res["bandwidth"])

    # E0  verification against the NumPy/FFT reference
    import reference
    u, v, _ = reference.run(64)
    ver = {}
    for prec, tol in (("f64", 1e-13), ("f32", 1e-5)):
        f = OUT / f"_ver_{prec}.bin"
        call("tg", "--N", 64, "--prec", prec, "--tol", tol, "--dump", f)
        g = np.fromfile(f); f.unlink()
        gu, gv = g[:4096].reshape(64, 64), g[4096:].reshape(64, 64)
        ver[prec] = float(max(abs(gu - u).max(), abs(gv - v).max()))
    res["verification_vs_numpy_fft_N64"] = ver
    print("verification", ver)

    # E1  Taylor-Green grid convergence
    Ns = [32, 64, 128, 256, 512, 1024]
    conv = {}
    for prec in ("f64", "f32"):
        rows = [call("tg", "--N", n, "--prec", prec, "--nu", 0.01, "--T", 1.0) for n in Ns]
        conv[prec] = {"N": Ns, "linf": [r["linf"] for r in rows], "l2": [r["l2"] for r in rows],
                      "order_linf": orders(Ns, [r["linf"] for r in rows]), "order_l2": orders(Ns, [r["l2"] for r in rows]),
                      "max_div": [r["max_div"] for r in rows], "steps": [r["steps"] for r in rows],
                      "wall_ms": [r["wall_ms"] for r in rows], "mg_cycles_per_solve": [r["mg_cycles_per_solve"] for r in rows]}
        print("convergence", prec, conv[prec]["linf"], conv[prec]["order_linf"], f"t={time.time()-t0:.0f}s")
    res["taylor_green"] = conv

    # E2  multigrid convergence vs N
    mg = {}
    for prec in ("f64", "f32"):
        mg[prec] = [call("mg", "--N", n, "--prec", prec) for n in [64, 128, 256, 512, 1024, 2048]]
        for r in mg[prec]:
            h = r["rel_resid"]
            r["factor_geomean_cycles_1_6"] = round((h[6] / h[0]) ** (1 / 6), 4)
    res["multigrid"] = mg
    print("multigrid", [(r["N"], r["factor_geomean_cycles_1_6"], r["ms_per_vcycle"]) for r in mg["f64"]])

    # E3  performance
    bench = {}
    for prec in ("f64", "f32"):
        bench[prec] = [call("bench", "--N", n, "--prec", prec, "--steps", 30) for n in [128, 256, 512, 1024, 2048]]
    res["bench"] = bench
    print("bench f32", [(r["N"], r["ms_per_step"], r["rhs_GBps"]) for r in bench["f32"]])
    print("bench f64", [(r["N"], r["ms_per_step"], r["rhs_GBps"]) for r in bench["f64"]])

    res["meta"]["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print("done", res["meta"]["runtime_s"], "s")


if __name__ == "__main__":
    main()
