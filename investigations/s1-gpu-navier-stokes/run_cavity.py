"""Systems study S1, milestone 3a: lid-driven cavity against Ghia, Ghia and Shin (1982).

    ./build.sh && python run_cavity.py        # writes results/cavity.json
"""
import json
import subprocess
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
EXE = str(HERE / "bin" / "cavity.exe")
OUT = HERE / "results"
GHIA = json.loads((HERE / "data" / "ghia1982.json").read_text())


def run(N, Re):
    dump = OUT / f"_cav_{N}_{Re}.bin"
    p = subprocess.run([EXE, "--N", str(N), "--Re", str(Re), "--tmax", "400", "--dump", str(dump)],
                       capture_output=True, text=True, check=True)
    r = json.loads(p.stdout.strip().replace("-nan(ind)", "NaN"))
    raw = np.fromfile(dump); dump.unlink()
    mu = (N + 1) * N
    u = raw[:mu].reshape(N, N + 1)            # [j, i]
    h = 1.0 / N
    # streamfunction at cell corners: psi(i, 0) = 0, d psi / dy = u
    psi = np.zeros((N + 1, N + 1))
    psi[1:, :] = np.cumsum(u * h, axis=0)
    jmin, imin = np.unravel_index(np.argmin(psi), psi.shape)
    r["psi_min"] = float(psi.min()); r["psi_min_xy"] = [imin * h, jmin * h]
    return r


def compare(r, Re):
    N = r["N"]; h = 1.0 / N
    gy = np.array(GHIA["u_centerline_x0.5"]["y"]); gu = np.array(GHIA["u_centerline_x0.5"][f"Re{Re}"])
    gx = np.array(GHIA["v_centerline_y0.5"]["x"]); gv = np.array(GHIA["v_centerline_y0.5"][f"Re{Re}"])
    yu = np.concatenate([[0.0], (np.arange(N) + 0.5) * h, [1.0]]); uu = np.concatenate([[0.0], r["u_x05"], [1.0]])
    xv = np.concatenate([[0.0], (np.arange(N) + 0.5) * h, [1.0]]); vv = np.concatenate([[0.0], r["v_y05"], [0.0]])
    mu = np.interp(gy, yu, uu); mv = np.interp(gx, xv, vv)
    inner_u = (gy > 0) & (gy < 1); inner_v = (gx > 0) & (gx < 1)     # exclude the imposed wall values
    du, dv = (mu - gu)[inner_u], (mv - gv)[inner_v]
    return {"u_at_ghia_points": mu.round(5).tolist(), "v_at_ghia_points": mv.round(5).tolist(),
            "u_max_abs_dev": float(abs(du).max()), "u_rms_dev": float(np.sqrt((du ** 2).mean())),
            "v_max_abs_dev": float(abs(dv).max()), "v_rms_dev": float(np.sqrt((dv ** 2).mean()))}


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time(); res = {"reference": GHIA["reference"], "cases": []}
    for Re in (100, 1000):
        for N in (64, 128, 256):
            r = run(N, Re)
            c = compare(r, Re)
            row = {k: r[k] for k in ("N", "Re", "dt", "t", "steps", "steady_rate", "converged", "max_div",
                                     "mg_cycles_per_solve", "wall_s", "psi_min", "psi_min_xy", "rate_history")}
            row.update(c); row["u_x05"] = r["u_x05"]; row["v_y05"] = r["v_y05"]
            res["cases"].append(row)
            print(f"Re={Re} N={N} converged={r['converged']} t={r['t']:.1f} wall={r['wall_s']:.0f}s "
                  f"u dev max {c['u_max_abs_dev']:.4f} rms {c['u_rms_dev']:.4f} | v dev max {c['v_max_abs_dev']:.4f} "
                  f"rms {c['v_rms_dev']:.4f} | psi_min {r['psi_min']:.6f} at {r['psi_min_xy']}  [{time.time()-t0:.0f}s]", flush=True)
            (OUT / "cavity.json").write_text(json.dumps(res))
    res["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "cavity.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
