"""Copy the numbers the website renders from results/ into site/data/inv02.json. Never edited by hand.

    python export_site_data.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
SITE = HERE.parents[1] / "site" / "data"
R = HERE / "results"
m1 = json.loads((R / "summary.json").read_text())
m2 = json.loads((R / "milestone2.json").read_text())
m3 = json.loads((R / "milestone3_perf.json").read_text())
cav = json.loads((R / "cavity.json").read_text())
ghia = json.loads((HERE / "data" / "ghia1982.json").read_text())

out = {
    "bandwidth": m1["bandwidth"],
    "verification": m1["verification_vs_numpy_fft_N64"],
    "convergence": {p: {k: m1["taylor_green"][p][k] for k in ("N", "linf", "order_linf", "max_div")} for p in ("f64", "f32")},
    "mg_factor": {p: [[r["N"], r["factor_geomean_cycles_1_6"]] for r in m1["multigrid"][p]] for p in ("f64", "f32")},
    "m1_bench_f64": [{k: r[k] for k in ("N", "ms_per_step", "phase_ms_per_step", "rhs_GBps")} for r in m1["bench"]["f64"]],
    "m1_bench_f32": [{k: r[k] for k in ("N", "ms_per_step", "mg_cycles_per_solve", "rhs_GBps")} for r in m1["bench"]["f32"]],
    "residual_history_N1024": m2["residual_history_N1024"],
    "ablation": {n: [{"name": x["name"], "ms": x["ms_per_step_median"], "lo": x["ms_per_step_min"], "hi": x["ms_per_step_max"],
                      "syncs": x["host_syncs_per_step"], "cycles": x["mg_cycles_per_solve"]} for x in rows]
                 for n, rows in m2["ablation"].items()},
    "vcycle_ms_m2": m2["vcycle_ms"],
    "tg_m2": m2["taylor_green"],
    "throughput": {
        "N": [128, 256, 512, 1024, 2048],
        "f64 (milestone 1)": [r["ms_per_step"] for r in m1["bench"]["f64"]],
        "f64": [m3["bench"]["f64_tb0"][str(n)] for n in (128, 256, 512, 1024, 2048)],
        "mixed": [m3["bench"]["mixed_tb0"][str(n)] for n in (128, 256, 512, 1024, 2048)],
        "mixed + blocked smoother": [m3["bench"]["mixed_tb1"][str(n)] for n in (128, 256, 512, 1024, 2048)],
        "f64 + blocked smoother": [m3["bench"]["f64_tb1"][str(n)] for n in (128, 256, 512, 1024, 2048)],
    },
    "cavity": [{k: c[k] for k in ("N", "Re", "steps", "t", "converged", "max_div", "u_max_abs_dev", "u_rms_dev",
                                  "v_max_abs_dev", "v_rms_dev", "psi_min", "psi_min_xy", "u_x05", "v_y05")} for c in cav["cases"]],
    "ghia": {k: ghia[k] for k in ("u_centerline_x0.5", "v_centerline_y0.5", "reference")},
}
roof = json.loads((R / "roofline.json").read_text())
out["roofline"] = {"N": roof["N"], "ceilings": roof["ceilings"],
                   "kernels": [{k: x.get(k) for k in ("kernel", "prec", "grid", "time_us", "dram_bytes", "GBps", "GFLOPs64", "GFLOPs32",
                                                  "intensity64", "intensity32", "dram_pct", "fp64_pipe_pct_elapsed", "lanes_per_inst", "fp64_lanes",
                                                  "occupancy_pct")} for x in roof["kernels"]]}
tbc = json.loads((R / "tb_compact.json").read_text())
out["tb_compact"] = {k: {kk: v[kk] for kk in ("branch", "compact", "speedup")} for k, v in tbc["vcycle_ms"].items()}
out["tb_step"] = {k: {kk: v[kk] for kk in ("branch", "compact", "speedup")} for k, v in tbc["step_ms"].items()}
SITE.mkdir(parents=True, exist_ok=True)
(SITE / "inv02.json").write_text(json.dumps(out, separators=(",", ":")))
print("wrote", SITE / "inv02.json", round((SITE / "inv02.json").stat().st_size / 1024, 1), "KB")
